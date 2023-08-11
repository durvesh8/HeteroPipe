import csv
import time
import subprocess
from ping3 import ping
import json

# Replace these with the actual IP addresses of your servers
servers = ["100.71.82.128", "100.71.82.16", "100.71.82.238"]

# Define the CSV file and the headers
filename = "ping_results.csv"
headers = ["timestamp", "server1", "server2", "latency", "packet_loss", "bandwidth"]

# Create the CSV file and write the headers
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

# Function to calculate packet loss
def calculate_packet_loss(dest, count=4):
    lost_packets = 0
    for i in range(count):
        if ping(dest) is None:
            lost_packets += 1
    return lost_packets / count

# Function to measure bandwidth
def measure_bandwidth(server):
    result = subprocess.run(['iperf3', '-c', server, '-J'], capture_output=True, text=True)
    json_output = json.loads(result.stdout)
    bandwidth = json_output['end']['sum_received']['bits_per_second']
    return bandwidth

# Get the current time and add 5 hours to it
end_time = time.time() + 12*60*60

while time.time() < end_time:
    for i in range(len(servers)):
        for j in range(i+1, len(servers)):
            # Ping from server i to server j
            latency = ping(servers[j])
            packet_loss = calculate_packet_loss(servers[j])
            bandwidth = measure_bandwidth(servers[j])

            # Write the results to the CSV file
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time.ctime(), servers[i], servers[j], latency, packet_loss, bandwidth])

    # Sleep for a while before the next round of pings
    time.sleep(60)
