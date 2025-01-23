from gradio_client import Client

client = Client("tiiuae/falcon-mamba-playground")
result = client.predict(
		message="Hello!!",
		temperature=0.3,
		max_new_tokens=1024,
		top_p=1,
		top_k=20,
		penalty=1.2,
		api_name="/chat"
)
print(result)