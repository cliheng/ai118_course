import chromadb

client = chromadb.HttpClient(host="localhost", port="8000")

print(client.list_collections())