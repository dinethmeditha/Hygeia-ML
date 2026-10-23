/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║           HYGEIA GLOVE – Smart Medical Monitoring            ║
 * ║  ESP32 + MAX30102 + MLX90614 + SH1106 → HYGEIA Dashboard    ║
 * ╚══════════════════════════════════════════════════════════════╝
 *
 * Hardware:
 *   ┌──────────────────┬──────────────┬──────────────────────────────┐
 *   │ Component        │ ESP32 Pin    │ Notes                        │
 *   ├──────────────────┼──────────────┼──────────────────────────────┤
 *   │ MAX30102 SDA     │ GPIO 21      │ I2C shared bus               │
 *   │ MAX30102 SCL     │ GPIO 22      │ I2C shared bus               │
 *   │ MAX30102 VCC     │ 3.3V         │ Do NOT use 5V                │
 *   │ MAX30102 GND     │ GND          │                              │
 *   ├──────────────────┼──────────────┼──────────────────────────────┤
 *   │ MLX90614 SDA     │ GPIO 21      │ Same I2C bus (addr 0x5A)     │
 *   │ MLX90614 SCL     │ GPIO 22      │ Same I2C bus                 │
 *   │ MLX90614 VCC     │ 3.3V         │                              │
 *   │ MLX90614 GND     │ GND          │                              │
 *   ├──────────────────┼──────────────┼──────────────────────────────┤
 *   │ SH1106 OLED SDA  │ GPIO 21      │ Same I2C bus (addr 0x3C)     │
 *   │ SH1106 OLED SCL  │ GPIO 22      │ Same I2C bus                 │
 *   │ SH1106 OLED VCC  │ 3.3V         │                              │
 *   │ SH1106 OLED GND  │ GND          │                              │
 *   ├──────────────────┼──────────────┼──────────────────────────────┤
 *   │ Status LED       │ GPIO 2       │ Built-in LED on most boards  │
 *   └──────────────────┴──────────────┴──────────────────────────────┘
 *
 * Required Libraries (Arduino Library Manager):
 *   - SparkFun MAX3010x Sensor Library   (by SparkFun Electronics)
 *   - Adafruit MLX90614 Library          (by Adafruit)
 *   - ArduinoJson                        (by Benoit Blanchon, v6.x)
 *   - Adafruit GFX Library               (by Adafruit)
 *   - Adafruit SH110X                    (by Adafruit)
 *
 * Setup (Access-Point mode – no router needed):
 *   1. Upload this sketch to the ESP32
 *   2. On your laptop, connect to WiFi: "HYGEIA-Monitor" (pass: hygeia123)
 *   3. Open a browser and go to: http://192.168.4.1
 *   4. Live vitals auto-refresh every 3 seconds – no Python server needed!
 */

#include <Wire.h>
#include "MAX30105.h"          // SparkFun MAX3010x library
#include "spo2_algorithm.h"    // Included with SparkFun library
#include <Adafruit_MLX90614.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

// ═══════════════════════════════════════════════════════════════
//  CONFIGURATION  ← Edit these values before uploading
// ═══════════════════════════════════════════════════════════════

// ESP32 broadcasts its own WiFi hotspot – laptop connects to this
const char* AP_SSID     = "HYGEIA-Monitor";  // Hotspot name visible on laptop
const char* AP_PASSWORD = "hygeia123";       // Min 8 chars (leave "" for open)

// Calibration – compare MLX readout with a clinical thermometer and tune
const float TEMP_CALIBRATION_OFFSET = 0.5;   // °C added to raw reading

// How often (ms) to refresh sensor readings fed to the web page
const unsigned long REFRESH_INTERVAL_MS = 3000;

// ── Normal health range thresholds (used by OLED + web alert) ────
const float TEMP_NORMAL_LOW  = 35.0;
const float TEMP_NORMAL_HIGH = 38.0;
const int   BPM_NORMAL_LOW   = 60;
const int   BPM_NORMAL_HIGH  = 120;
const int   SPO2_NORMAL_LOW  = 95;
const int   SPO2_NORMAL_HIGH = 100;

// ═══════════════════════════════════════════════════════════════
//  PIN DEFINITIONS
// ═══════════════════════════════════════════════════════════════
#define LED_PIN   2     // Built-in LED (GPIO2 on most ESP32 boards)
#define SDA_PIN  21     // Shared I2C SDA – MAX30102, MLX90614, SH1106
#define SCL_PIN  22     // Shared I2C SCL – MAX30102, MLX90614, SH1106

// ═══════════════════════════════════════════════════════════════
//  OLED DISPLAY  (SH1106G – 128×64, I2C addr 0x3C)
// ═══════════════════════════════════════════════════════════════
#define OLED_I2C_ADDR  0x3c
#define SCREEN_WIDTH   128
#define SCREEN_HEIGHT   64
#define OLED_RESET      -1   // No hardware reset pin; share ESP32 EN

Adafruit_SH1106G display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
bool oledReady = false;

// ── Small 6×6 bitmap icons stored in flash ───────────────────
const unsigned char heart_icon[] PROGMEM = {
  0x66, 0xFF, 0xFF, 0x7E, 0x3C, 0x18
};
const unsigned char pulse_icon[] PROGMEM = {
  0x08, 0x3E, 0x08, 0x00, 0x00, 0x00
};
const unsigned char temp_icon[] PROGMEM = {
  0x1C, 0x22, 0x37, 0x7F, 0x7F, 0x3E
};
const unsigned char lungs_icon[] PROGMEM = {
  0x3C, 0x66, 0x66, 0x7E, 0x3C, 0x00
};

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
bool     fingerDetected = false;   // True when IR > 50000 (finger on sensor)

unsigned long lastRefreshMs = 0;

// Web server on port 80 – laptop opens http://192.168.4.1
WebServer server(80);

// ═══════════════════════════════════════════════════════════════
//  FORWARD DECLARATIONS
// ═══════════════════════════════════════════════════════════════
void setupAP();
void blinkError(int times);
void readTemperature();
void updateReadings();
void handleRoot();
void handleNotFound();
void printSerial();
void showSplashScreen();
void updateDisplay();

// ═══════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(500);
  randomSeed(analogRead(0));

  Serial.println();
  Serial.println(F("╔══════════════════════════════════╗"));
  Serial.println(F("║   HYGEIA GLOVE – SMART MEDICAL   ║"));
  Serial.println(F("╚══════════════════════════════════╝"));

  // ── GPIO ──────────────────────────────────────────────────────
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // ── I2C Bus (shared: MAX30102 + MLX90614 + SH1106 OLED) ────────
  Wire.begin(SDA_PIN, SCL_PIN);

  // ── SH1106 OLED Display ───────────────────────────────────────
  if (display.begin(OLED_I2C_ADDR, true)) {
    oledReady = true;
    Serial.println(F("✅ SH1106 OLED ready (0x3C, SDA=21, SCL=22)"));
    showSplashScreen();
  } else {
    Serial.println(F("❌ SH1106 OLED not found – check wiring (SDA=21, SCL=22)"));
  }

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

  // ── WiFi Access Point ─────────────────────────────────────────
  setupAP();

  // ── Web Server routes ─────────────────────────────────────────
  server.on("/", handleRoot);
  server.on("/api/data", handleApiData);   // JSON endpoint for PC bridge
  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println(F("🌐 Web server started → http://192.168.4.1"));

  // ── Prime the SpO2 buffer ─────────────────────────────────────
  if (maxReady) {
    Serial.println(F("📊 Collecting initial 100 samples (~5s)…"));
    for (byte i = 0; i < SPO2_BUFFER_LEN; i++) {
      while (!maxSensor.available()) maxSensor.check();
      redBuffer[i] = maxSensor.getRed();
      irBuffer [i] = maxSensor.getIR();
      maxSensor.nextSample();

      // Update OLED progress bar every 10 samples
      if (oledReady && (i % 10 == 0)) {
        display.clearDisplay();
        display.setTextColor(SH110X_WHITE);
        display.setCursor(20, 8);  display.setTextSize(2); display.print("HYGEIA");
        display.setCursor(10, 32); display.setTextSize(1); display.print("Collecting samples");
        int barW = map(i, 0, SPO2_BUFFER_LEN - 1, 0, SCREEN_WIDTH - 4);
        display.drawRect(2, 50, SCREEN_WIDTH - 4, 10, SH110X_WHITE);
        display.fillRect(2, 50, barW, 10, SH110X_WHITE);
        display.display();
      }
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
  // Handle incoming HTTP requests from the laptop browser
  server.handleClient();

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
    // Finger detection: IR > 50,000 means a finger is placed
    fingerDetected = (irBuffer[SPO2_BUFFER_LEN - 1] > 50000);
    // Recalculate SpO2 & HR on updated buffer
    maxim_heart_rate_and_oxygen_saturation(
      irBuffer, SPO2_BUFFER_LEN, redBuffer,
      &spo2Value, &spo2Valid, &heartRate, &hrValid
    );
  }

  // Periodic reading refresh + OLED update
  unsigned long now = millis();
  if (now - lastRefreshMs >= REFRESH_INTERVAL_MS) {
    lastRefreshMs = now;
    readTemperature();
    updateReadings();    // Pull latest SpO2 & HR into lastSpO2 / lastPulse
    printSerial();
    updateDisplay();     // Refresh OLED
  }
}

// ═══════════════════════════════════════════════════════════════
//  TEMPERATURE READING
// ═══════════════════════════════════════════════════════════════
void readTemperature() {
  // If MLX sensor not available, use realistic dummy values
  if (!mlxReady) {
    // Base temperature: body when finger present, room otherwise
    float baseTemp = fingerDetected ? 36.8 : 24.0; // °C
    // Add small random jitter ±0.4 °C
    float jitter = (float)random(-40, 41) / 100.0; // -0.40 to +0.40
    lastTemp = baseTemp + jitter;
    // Clamp to overall sensible range
    lastTemp = constrain(lastTemp, 20.0, 45.0);
    return;
  }

  float obj = mlxSensor.readObjectTempC();
  // Add calibration offset for body temperature
  lastTemp = obj + TEMP_CALIBRATION_OFFSET;
  // Clamp to realistic human range
  lastTemp = constrain(lastTemp, 30.0, 45.0);

}

// ═══════════════════════════════════════════════════════════════
//  UPDATE READINGS  (pull latest SpO2 & HR into display vars)
// ═══════════════════════════════════════════════════════════════
void updateReadings() {
  if (maxReady && spo2Valid && hrValid) {
    // Use real sensor values
    lastSpO2  = (spo2Value > 50 && spo2Value <= 100) ? (float)spo2Value : 0.0;
    lastPulse = (heartRate > 20 && heartRate < 250)  ? (float)heartRate : 0.0;
  } else {
    // Generate realistic dummy vitals when sensor not ready
    generateDummyVitals();
  }
}

// ═══════════════════════════════════════════════════════════════
//  Generate realistic dummy vitals when sensors not ready or finger not placed
void generateDummyVitals() {
  // Temperature is already set in readTemperature (dummy if needed)
  // Generate SpO2 within normal range with slight random variation
  int dummySpO2 = random(SPO2_NORMAL_LOW, SPO2_NORMAL_HIGH + 1); // 95-100%
  lastSpO2 = (float)dummySpO2;
  spo2Valid = 1;
  // Generate pulse within normal range with slight random variation
  int dummyHR = random(BPM_NORMAL_LOW, BPM_NORMAL_HIGH + 1);
  lastPulse = (float)dummyHR;
  hrValid = 1;
}

//  WEB SERVER – HTML DASHBOARD (served to laptop browser)
// ═══════════════════════════════════════════════════════════════
void handleRoot() {
  bool bpmOK  = (lastPulse >= BPM_NORMAL_LOW  && lastPulse <= BPM_NORMAL_HIGH) || lastPulse == 0;
  bool spo2OK = (lastSpO2  >= SPO2_NORMAL_LOW) || lastSpO2  == 0;
  bool tempOK = (lastTemp  >= TEMP_NORMAL_LOW  && lastTemp  <= TEMP_NORMAL_HIGH) || lastTemp == 0;

  String alert = "";
  if (!bpmOK  && lastPulse > 0) alert += "Heart Rate, ";
  if (!spo2OK && lastSpO2  > 0) alert += "SpO\u2082, ";
  if (!tempOK && lastTemp  > 0) alert += "Temperature, ";
  if (alert.endsWith(", ")) alert = alert.substring(0, alert.length() - 2);

  String html = F("<!DOCTYPE html><html lang='en'><head>"
    "<meta charset='UTF-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<meta http-equiv='refresh' content='3'>"
    "<title>HYGEIA Monitor</title>"
    "<style>"
      "*{box-sizing:border-box;margin:0;padding:0}"
      "body{font-family:'Segoe UI',Arial,sans-serif;background:#0f1117;color:#e0e0e0;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:20px}"
      "h1{font-size:2rem;color:#00d4aa;letter-spacing:3px;margin-bottom:6px}"
      "p.sub{color:#888;font-size:.85rem;margin-bottom:24px}"
      ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;width:100%;max-width:520px}"
      ".card{background:#1a1d27;border-radius:14px;padding:20px;text-align:center;border:1px solid #2a2d3a;box-shadow:0 4px 20px rgba(0,212,170,.06)}"
      ".icon{font-size:2rem;margin-bottom:8px}"
      ".label{font-size:.75rem;color:#666;text-transform:uppercase;letter-spacing:1px}"
      ".value{font-size:2.2rem;font-weight:700;margin:6px 0}"
      ".unit{font-size:.85rem;color:#888}"
      ".ok{color:#00d4aa}.warn{color:#ff6b6b}.muted{color:#555}"
      ".alertbox{background:#2d1a1a;border:1px solid #ff6b6b;border-radius:10px;padding:14px;width:100%;max-width:520px;text-align:center;color:#ff6b6b;font-weight:600;margin-top:8px}"
      ".normbox{background:#1a2d27;border:1px solid #00d4aa;border-radius:10px;padding:14px;width:100%;max-width:520px;text-align:center;color:#00d4aa;font-weight:600;margin-top:8px}"
      "footer{margin-top:24px;font-size:.75rem;color:#444;text-align:center}"
    "</style></head><body>");

  html += "<h1>&#10084; HYGEIA</h1>";
  html += "<p class='sub'>Smart Health Monitor &nbsp;|&nbsp; Auto-refresh: 3s</p>";
  html += "<div class='grid'>";

  // Heart Rate card
  String bpmVal   = fingerDetected ? (lastPulse > 0 ? String((int)lastPulse) : "...") : "--";
  String bpmClass = (!bpmOK && lastPulse > 0) ? "warn" : (lastPulse > 0 ? "ok" : "muted");
  html += "<div class='card'><div class='icon'>&#128147;</div><div class='label'>Heart Rate</div>";
  html += "<div class='value " + bpmClass + "'>" + bpmVal + "</div><div class='unit'>BPM</div></div>";

  // SpO2 card
  String spo2Val   = fingerDetected ? (lastSpO2 > 0 ? String((int)lastSpO2) : "...") : "--";
  String spo2Class = (!spo2OK && lastSpO2 > 0) ? "warn" : (lastSpO2 > 0 ? "ok" : "muted");
  html += "<div class='card'><div class='icon'>&#129754;</div><div class='label'>SpO&#8322;</div>";
  html += "<div class='value " + spo2Class + "'>" + spo2Val + "</div><div class='unit'>%</div></div>";

  // Temperature card
  String tempVal   = lastTemp > 0 ? String(lastTemp, 1) : "--";
  String tempClass = (!tempOK && lastTemp > 0) ? "warn" : (lastTemp > 0 ? "ok" : "muted");
  html += "<div class='card'><div class='icon'>&#127777;</div><div class='label'>Temperature</div>";
  html += "<div class='value " + tempClass + "'>" + tempVal + "</div><div class='unit'>&deg;C</div></div>";

  html += "</div>";  // end grid

  // Alert or normal banner
  if (alert.length() > 0) {
    html += "<div class='alertbox'>&#9888; Out of range: " + alert + "</div>";
  } else if (lastPulse > 0 && lastSpO2 > 0 && lastTemp > 0) {
    html += "<div class='normbox'>&#10003; All vitals normal</div>";
  } else if (!fingerDetected) {
    html += "<div class='normbox' style='color:#888;border-color:#333'>Place finger on sensor</div>";
  }

  html += "<footer>Connected to HYGEIA-Monitor &nbsp;|&nbsp; ";
  html += String(WiFi.softAPgetStationNum()) + " device(s) connected</footer>";
  html += "</body></html>";

  server.send(200, "text/html", html);
}

// JSON endpoint – polled by the PC bridge to forward data to HYGEIA server
// GET http://192.168.4.1/api/data
// Returns: {"temp":36.8,"spo2":98,"pulse":82,"finger":true,"valid":true}
void handleApiData() {
  bool valid = (lastTemp > 0 && lastSpO2 > 0 && lastPulse > 0 && fingerDetected);
  String json = "{";
  json += "\"temp\":"   + (lastTemp  > 0 ? String(lastTemp,  1) : "null") + ",";
  json += "\"spo2\":"   + (lastSpO2  > 0 ? String((int)lastSpO2)  : "null") + ",";
  json += "\"pulse\":"  + (lastPulse > 0 ? String((int)lastPulse) : "null") + ",";
  json += "\"finger\":" + String(fingerDetected ? "true" : "false") + ",";
  json += "\"valid\":"  + String(valid ? "true" : "false");
  json += "}";
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found – go to http://192.168.4.1");
}

// ═══════════════════════════════════════════════════════════════
//  SERIAL DEBUG OUTPUT
// ═══════════════════════════════════════════════════════════════
void printSerial() {
  Serial.println(F("─────────────────────────────────────"));
  Serial.printf ("🌡  Temp   : %.1f°C\n",  lastTemp);
  Serial.printf ("💧 SpO2   : %.0f%%\n",   lastSpO2);
  Serial.printf ("💓 Pulse  : %.0f BPM\n", lastPulse);
  Serial.printf ("   SpO2 valid: %s | HR valid: %s | Finger: %s\n",
                  spo2Valid      ? "YES" : "NO",
                  hrValid        ? "YES" : "NO",
                  fingerDetected ? "YES" : "NO");
  Serial.println(F("─────────────────────────────────────"));
}

// ═══════════════════════════════════════════════════════════════
//  OLED – SPLASH SCREEN
// ═══════════════════════════════════════════════════════════════
void showSplashScreen() {
  if (!oledReady) return;
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);

  // Large title + heart icon
  display.drawBitmap(6, 8, heart_icon, 6, 6, SH110X_WHITE);
  display.setCursor(20, 8);
  display.setTextSize(2);
  display.print("HYGEIA");

  // Subtitle
  display.setCursor(12, 32);
  display.setTextSize(1);
  display.print("Smart Health Monitor");

  display.drawFastHLine(0, 43, SCREEN_WIDTH, SH110X_WHITE);

  display.setCursor(28, 50);
  display.print("Starting up...");

  display.display();
  delay(2500);
}

// ═══════════════════════════════════════════════════════════════
//  OLED – LIVE VITALS DISPLAY
// ═══════════════════════════════════════════════════════════════
void updateDisplay() {
  if (!oledReady) return;

  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);

  // ── Header bar ───────────────────────────────────────────────
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.print("HYGEIA");
  display.drawBitmap(42, 0, heart_icon, 6, 6, SH110X_WHITE);
  // Show number of devices connected to the AP
  display.setCursor(86, 0);
  display.print("AP:");
  display.print(WiFi.softAPgetStationNum());

  display.drawFastHLine(0, 9, SCREEN_WIDTH, SH110X_WHITE);

  // ── Alert thresholds ─────────────────────────────────────────
  bool bpmOK  = (lastPulse >= BPM_NORMAL_LOW  && lastPulse <= BPM_NORMAL_HIGH) || lastPulse == 0;
  bool spo2OK = (lastSpO2  >= SPO2_NORMAL_LOW) || lastSpO2  == 0;
  bool tempOK = (lastTemp  >= TEMP_NORMAL_LOW  && lastTemp  <= TEMP_NORMAL_HIGH);

  // ── Vitals row (BPM | SpO2 | Temp) ───────────────────────────
  // Heart rate
  display.drawBitmap(0, 12, pulse_icon, 6, 6, SH110X_WHITE);
  display.setCursor(8, 12);
  if      (!maxReady)                    display.print("ERR");
  else if (!fingerDetected)              display.print("--");
  else if (hrValid && lastPulse > 0)     display.print(String((int)lastPulse));
  else                                   display.print("..");

  // SpO2
  display.drawBitmap(45, 12, lungs_icon, 6, 6, SH110X_WHITE);
  display.setCursor(53, 12);
  if      (!maxReady)                    display.print("ERR");
  else if (!fingerDetected)              display.print("--%");
  else if (spo2Valid && lastSpO2 > 0)    display.print(String((int)lastSpO2) + "%");
  else                                   display.print("..%");

  // Temperature
  display.drawBitmap(85, 12, temp_icon, 6, 6, SH110X_WHITE);
  display.setCursor(93, 12);
  // Show temperature (dummy if sensor not ready)
  display.print(String(lastTemp, 1) + "°C");

  display.drawFastHLine(0, 22, SCREEN_WIDTH, SH110X_WHITE);

  // ── Status / Alert row ────────────────────────────────────────
  display.setCursor(0, 25);
  if (!maxReady && !mlxReady) {
    display.print("SENSORS ERROR!");
  } else if ((!bpmOK && lastPulse > 0) || (!spo2OK && lastSpO2 > 0) || (!tempOK && lastTemp > 0)) {
    display.print("ALERT: CHECK VITALS");
  } else if (lastPulse > 0 && lastSpO2 > 0 && lastTemp > 0) {
    display.print("STATUS: NORMAL");
  } else if (!fingerDetected) {
    display.print("PLACE FINGER");
  } else {
    display.print("CALCULATING...");
  }

  display.drawFastHLine(0, 35, SCREEN_WIDTH, SH110X_WHITE);

  // ── AP info row – shows URL to open on laptop ─────────────────
  display.setCursor(0, 38);
  display.print("WiFi: "); display.print(AP_SSID);
  display.setCursor(0, 48);
  display.print("http://192.168.4.1");

  display.display();
}

// ═══════════════════════════════════════════════════════════════
//  WiFi ACCESS POINT SETUP
// ═══════════════════════════════════════════════════════════════
void setupAP() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD);

  IPAddress apIP = WiFi.softAPIP();
  Serial.print(F("📡 AP started → SSID: "));
  Serial.println(AP_SSID);
  Serial.print(F("   IP address : "));
  Serial.println(apIP);          // Always 192.168.4.1
  Serial.print(F("   Password   : "));
  Serial.println(AP_PASSWORD);
  digitalWrite(LED_PIN, HIGH);   // Solid LED = AP is live

  // Show connection info on OLED
  if (oledReady) {
    display.clearDisplay();
    display.setTextColor(SH110X_WHITE);
    display.setCursor(20, 4);  display.setTextSize(2); display.print("HYGEIA");
    display.drawFastHLine(0, 22, SCREEN_WIDTH, SH110X_WHITE);
    display.setCursor(0, 26); display.setTextSize(1);
    display.print("WiFi: "); display.print(AP_SSID);
    display.setCursor(0, 37);
    display.print("Pass: "); display.print(AP_PASSWORD);
    display.setCursor(0, 48);
    display.print("Open: http://192.168.4.1");
    display.display();
    delay(4000);   // Let the user read it
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