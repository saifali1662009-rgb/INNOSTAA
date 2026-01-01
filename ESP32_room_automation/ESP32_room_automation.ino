#include <WiFi.h>

const char* ssid = "Harshit";
const char* password = "12345678";

WiFiServer server(80);

int ledPin = 2; // Use GPIO2 or any pin you like

void setup() {
  Serial.begin(115200);
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);

  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();
}

void loop() {
  WiFiClient client = server.available();
  if (!client) return;

  String req = client.readStringUntil('\r');
  client.flush();

  // TURN LIGHT ON
  if (req.indexOf("/light/on") != -1) {
    digitalWrite(ledPin, HIGH);
    Serial.println("Light ON");
  }

  // TURN LIGHT OFF
  else if (req.indexOf("/light/off") != -1) {
    digitalWrite(ledPin, LOW);
    Serial.println("Light OFF");
  }

  // PING CHECK
  else if (req.indexOf("/ping") != -1) {
    Serial.println("Ping received");
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: text/plain");
    client.println("Connection: close");
    client.println();
    client.println("pong");
    client.println();
    return;
  }

  // DEFAULT RESPONSE
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/html");
  client.println("Connection: close");
  client.println();
  client.println("OK");
  client.println();
}

