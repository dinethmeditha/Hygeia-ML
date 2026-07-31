/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║           HYGEIA GLOVE – Smart Medical Monitoring            ║
 * ║       ESP32 + MAX30102 + MLX90614 → HYGEIA Dashboard         ║
 * ╚══════════════════════════════════════════════════════════════╝
 *
 * Hardware:
 *   ┌──────────────┬──────────────┬──────────────────────────────┐
 *   │ Component    │ ESP32 Pin    │ Notes                        │
 *   ├──────────────┼──────────────┼──────────────────────────────┤
 *   │ MAX30102 SDA │ GPIO 21      │ I2C shared bus               │
 *   │ MAX30102 SCL │ GPIO 22      │ I2C shared bus               │
 *   │ MAX30102 VCC │ 3.3V         │ Do NOT use 5V                │
 *   │ MAX30102 GND │ GND          │                              │
 *   ├──────────────┼──────────────┼──────────────────────────────┤
 *   │ MLX90614 SDA │ GPIO 21      │ Same I2C bus (addr 0x5A)     │
 *   │ MLX90614 SCL │ GPIO 22      │ Same I2C bus                 │
 *   │ MLX90614 VCC │ 3.3V         │                              │
 *   │ MLX90614 GND │ GND          │                              │
 *   ├──────────────┼──────────────┼──────────────────────────────┤
 *   │ Status LED   │ GPIO 2       │ Built-in LED on most boards  │
 *   └──────────────┴──────────────┴──────────────────────────────┘
 *
 * Required Libraries (Arduino Library Manager):
 *   - SparkFun MAX3010x Sensor Library   (by SparkFun Electronics)
 *   - Adafruit MLX90614 Library          (by Adafruit)
 *   - ArduinoJson                        (by Benoit Blanchon, v6.x)
 *
 * Setup:
 *   1. Set WIFI_SSID / WIFI_PASSWORD
 *   2. Set SERVER_IP to your PC's local IP (run `ipconfig` on Windows)
 *   3. Run `python server.py` on your PC
 *   4. Upload this sketch to ESP32
 */

#include <Wire.h>
#include "MAX30105.h"          // SparkFun MAX3010x library
#include "spo2_algorithm.h"    // Included with SparkFun library
#include <Adafruit_MLX90614.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ═══════════════════════════════════════════════════════════════
//  CONFIGURATION  ← Edit these values before uploading
// ═══════════════════════════════════════════════════════════════
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Your PC's local IP address (run `ipconfig` on Windows → IPv4 Address)
const char* SERVER_HOST   = "192.168.1.100";
const int   SERVER_PORT   = 5000;
const char* ENDPOINT      = "/api/reading";

// Calibration – adjust offset based on your glove sensor placement
// Compare MLX readout with a clinical thermometer and tune this value
const float TEMP_CALIBRATION_OFFSET = 0.5;    // °C offset added to raw reading

// Send interval (milliseconds)
const unsigned long SEND_INTERVAL_MS = 5000;

// ═══════════════════════════════════════════════════════════════
//  PIN DEFINITIONS
// ═══════════════════════════════════════════════════════════════
#define LED_PIN        2     // Built-in LED (GPIO2 on most ESP32 boards)
#define SDA_PIN       21
#define SCL_PIN       22

// ═══════════════════════════════════════════════════════════════
//  SENSOR OBJECTS
// ═══════════════════════════════════════════════════════════════
MAX30105          maxSensor;
Adafruit_MLX90614 mlxSensor;

// ═══════════════════════════════════════════════════════════════
//  MAX30102 SpO2 BUFFER  (100 samples required by algorithm)
// ═══════════════════════════════════════════════════════════════
#define SPO2_BUFFER_LEN  100

uint32_t irBuffer [SPO2_BUFFER_LEN];
uint32_t redBuffer[SPO2_BUFFER_LEN];

int32_t  spo2Value  = 0;
int8_t   spo2Valid  = 0;
int32_t  heartRate  = 0;
int8_t   hrValid    = 0;

// ═══════════════════════════════════════════════════════════════
//  CURRENT READINGS
// ═══════════════════════════════════════════════════════════════
float    lastTemp    = 0.0;
float    lastSpO2    = 0.0;
float    lastPulse   = 0.0;
bool     maxReady    = false;
bool     mlxReady    = false;

unsigned long lastSendMs = 0;

// ═══════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println();
  Serial.println(F("╔══════════════════════════════════╗"));
  Serial.println(F("║   HYGEIA GLOVE – SMART MEDICAL   ║"));
  Serial.println(F("╚══════════════════════════════════╝"));

  // ── GPIO ──────────────────────────────────────────────────────
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // ── I2C Bus ───────────────────────────────────────────────────
  Wire.begin(SDA_PIN, SCL_PIN);

  // ── MLX90614 Temperature ──────────────────────────────────────
  if (mlxSensor.begin()) {
    mlxReady = true;
    Serial.println(F("✅ MLX90614 ready"));
    Serial.print  (F("   Emissivity: "));
    Serial.println(mlxSensor.readEmissivity());
  } else {
    Serial.println(F("❌ MLX90614 not found – check wiring (SDA=21, SCL=22)"));
    blinkError(5);
  }

  // ── MAX30102 Pulse Oximeter ────────────────────────────────────
  if (maxSensor.begin(Wire, I2C_SPEED_FAST)) {
    maxReady = true;
    Serial.println(F("✅ MAX30102 ready"));

    // Recommended settings for finger SpO2 measurement
    byte ledBrightness = 60;    // 0–255 (50 mA max)
    byte sampleAvg     = 4;     // Average over 4 samples per data point
    byte ledMode       = 2;     // Red + IR (needed for SpO2)
    int  sampleRate    = 100;   // 100 samples/sec
    int  pulseWidth    = 411;   // 411 µs → highest resolution
    int  adcRange      = 4096;  // 4096 nA full-scale

    maxSensor.setup(ledBrightness, sampleAvg, ledMode, sampleRate, pulseWidth, adcRange);
    maxSensor.setPulseAmplitudeRed(0x0A);   // Low Red LED for testing
    maxSensor.setPulseAmplitudeGreen(0);    // Green LED off

  } else {
    Serial.println(F("❌ MAX30102 not found – check wiring (SDA=21, SCL=22, VCC=3.3V!)"));
    blinkError(5);
  }

  // ── WiFi ──────────────────────────────────────────────────────
  connectWiFi();

  // ── Prime the SpO2 buffer ─────────────────────────────────────
  if (maxReady) {
    Serial.println(F("📊 Collecting initial 100 samples (~5s)…"));
    for (byte i = 0; i < SPO2_BUFFER_LEN; i++) {
      while (!maxSensor.available()) maxSensor.check();
      redBuffer[i] = maxSensor.getRed();
      irBuffer [i] = maxSensor.getIR();
      maxSensor.nextSample();
    }
    Serial.println(F("✅ Buffer ready"));
  }

  Serial.println(F("🟢 HYGEIA GLOVE is monitoring…"));
  Serial.println();
}

// ═══════════════════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════════════════
void loop() {
  // Keep WiFi alive
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println(F("⚠️  WiFi lost – reconnecting…"));
    connectWiFi();
  }

  // Slide the MAX30102 buffer: drop first 25, collect 25 new
  if (maxReady) {
    for (byte i = 25; i < SPO2_BUFFER_LEN; i++) {
      redBuffer[i - 25] = redBuffer[i];
      irBuffer [i - 25] = irBuffer [i];
    }
    for (byte i = 75; i < SPO2_BUFFER_LEN; i++) {
      while (!maxSensor.available()) maxSensor.check();
      redBuffer[i] = maxSensor.getRed();
      irBuffer [i] = maxSensor.getIR();
      maxSensor.nextSample();
    }
    // Recalculate SpO2 & HR on updated buffer
    maxim_heart_rate_and_oxygen_saturation(
      irBuffer, SPO2_BUFFER_LEN, redBuffer,
      &spo2Value, &spo2Valid, &heartRate, &hrValid
    );
  }

  // Timed send
  unsigned long now = millis();
  if (now - lastSendMs >= SEND_INTERVAL_MS) {
    lastSendMs = now;
    readTemperature();
    assembleSend();
    printSerial();
  }
}

// ═══════════════════════════════════════════════════════════════
//  TEMPERATURE READING
// ═══════════════════════════════════════════════════════════════
void readTemperature() {
  if (!mlxReady) { lastTemp = 0.0; return; }

  float obj = mlxSensor.readObjectTempC();
  // MLX90614 reads the object surface; add calibration offset for body temp
  lastTemp = obj + TEMP_CALIBRATION_OFFSET;
  lastTemp = constrain(lastTemp, 30.0, 45.0);
}

// ═══════════════════════════════════════════════════════════════
//  SEND TO HYGEIA SERVER
// ═══════════════════════════════════════════════════════════════
void assembleSend() {
  // Extract valid SpO2 & HR
  if (spo2Valid && spo2Value > 50 && spo2Value <= 100)
    lastSpO2 = (float)spo2Value;
  else
    lastSpO2 = 0.0;

  if (hrValid && heartRate > 20 && heartRate < 250)
    lastPulse = (float)heartRate;
  else
    lastPulse = 0.0;

  // Require all three readings to be valid
  if (lastTemp < 30.0 || lastTemp > 45.0 || lastSpO2 <= 0.0 || lastPulse <= 0.0) {
    Serial.println(F("⚠️  Sensor reading invalid – retrying next cycle"));
    return;
  }

  sendReading(lastTemp, lastSpO2, lastPulse);
}

void sendReading(float temp, float spo2, float pulse) {
  // Build JSON
  StaticJsonDocument<128> doc;
  doc["temp"]  = round(temp * 10.0) / 10.0;   // 1 decimal place
  doc["spo2"]  = (int)spo2;
  doc["pulse"] = (int)pulse;

  String payload;
  serializeJson(doc, payload);

  // Build URL
  char url[128];
  snprintf(url, sizeof(url), "http://%s:%d%s", SERVER_HOST, SERVER_PORT, ENDPOINT);

  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Device", "HYGEIA-ESP32");
  http.setTimeout(4000);

  int code = http.POST(payload);

  if (code == HTTP_CODE_OK) {
    Serial.print(F("📤 Sent OK → "));
    Serial.println(http.getString());
    // Confirmation flash
    digitalWrite(LED_PIN, LOW);  delay(80);
    digitalWrite(LED_PIN, HIGH);
  } else {
    Serial.print(F("❌ HTTP "));
    Serial.print(code);
    Serial.print(F(" – payload: "));
    Serial.println(payload);
  }

  http.end();
}

// ═══════════════════════════════════════════════════════════════
//  SERIAL DEBUG OUTPUT
// ═══════════════════════════════════════════════════════════════
void printSerial() {
  Serial.println(F("─────────────────────────────────────"));
  Serial.printf ("🌡  Temp   : %.1f°C\n",  lastTemp);
  Serial.printf ("💧 SpO2   : %.0f%%\n",   lastSpO2);
  Serial.printf ("💓 Pulse  : %.0f BPM\n", lastPulse);
  Serial.printf ("   SpO2 valid: %s | HR valid: %s\n",
                  spo2Valid ? "YES" : "NO",
                  hrValid   ? "YES" : "NO");
  Serial.println(F("─────────────────────────────────────"));
}

// ═══════════════════════════════════════════════════════════════
//  WiFi HELPER
// ═══════════════════════════════════════════════════════════════
void connectWiFi() {
  Serial.print(F("📶 Connecting to: "));
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(1000);
    Serial.print('.');
    attempts++;
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(F("✅ Connected! IP: "));
    Serial.println(WiFi.localIP());
    digitalWrite(LED_PIN, HIGH);
  } else {
    Serial.println(F("❌ WiFi failed – will retry in main loop"));
    digitalWrite(LED_PIN, LOW);
  }
}

// ═══════════════════════════════════════════════════════════════
//  ERROR BLINK
// ═══════════════════════════════════════════════════════════════
void blinkError(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH); delay(150);
    digitalWrite(LED_PIN, LOW);  delay(150);
  }
}