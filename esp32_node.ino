#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

const char* serverUrl = "https://smart-agriculture-gaza-ai.onrender.com/api/esp32";

#define RELAY_PIN 12
#define SOIL_PIN 34

WiFiClientSecure secureClient;

void setup() {
    Serial.begin(115200);
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW);

    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    secureClient.setInsecure();
    Serial.println("\nConnected to WiFi");
}

void loop() {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(secureClient, serverUrl);
        http.addHeader("Content-Type", "application/json");

        float temperature = 29.5;
        float pressure = 9984.5;
        int rawSoilReading = analogRead(SOIL_PIN);

      
        float soilMoisture = map(rawSoilReading, 0, 4095, 0, 480);

        String payload = "{\"temp\":" + String(temperature)
            + ",\"pressure\":" + String(pressure)
            + ",\"moisture\":" + String(soilMoisture) + "}";

        Serial.println("Sending: " + payload);
        int responseCode = http.POST(payload);

        if (responseCode > 0) {
            String response = http.getString();
            Serial.println("Server reply: " + response);

            StaticJsonDocument<256> document;
            DeserializationError error = deserializeJson(document, response);

            if (!error && document["success"] == true) {
                int relayState = document["relay_state"];
                const char* soilClass = document["class"];

                digitalWrite(RELAY_PIN, relayState == 1 ? HIGH : LOW);
                Serial.println("Class: " + String(soilClass));
                Serial.println(relayState == 1 ? "PUMP ON" : "PUMP OFF");
            } else {
                Serial.println("Invalid JSON response");
            }
        } else {
            Serial.println("Cannot reach the server");
        }

        http.end();
    }

    delay(5000);
}
