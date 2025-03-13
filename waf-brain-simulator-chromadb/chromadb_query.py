import chromadb

# Load the persisted ChromaDB database
client = chromadb.PersistentClient(path="./testchromaDB")

# Retrieve the collection by name (use the collection name you used when adding the data)
collection_name = "payloads"
collection = client.get_collection(name=collection_name)

# Get all items from the collection
# This returns a dictionary containing lists for "ids", "documents", "metadatas", and possibly "embeddings"
results = collection.get()

# Display the items
print("IDs:", results["ids"])
print("Documents:", results["documents"])
print("Metadata:", results["metadatas"])

# Optionally, iterate over all items to print details
for i, doc in enumerate(results["documents"]):
    print(f"\nItem {i}:")
    print(f"  ID: {results['ids'][i]}")
    print(f"  Document: {doc}")
    print(f"  Metadata: {results['metadatas'][i]}")