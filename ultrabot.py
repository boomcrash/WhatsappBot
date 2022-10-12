import http.client

conn = http.client.HTTPSConnection("api.ultramsg.com")

payload = "token=on8a6sdqqjnhl2iq&to=+593979602738&body=como estas el dia de hoy&priority=1&referenceId="

headers = { 'content-type': "application/x-www-form-urlencoded" }

conn.request("POST", "/instance19879/messages/chat", payload, headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))