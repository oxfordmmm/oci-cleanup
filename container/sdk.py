
from oci.artifacts import ArtifactsClient
from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(4), wait=wait_fixed(10))
def delete_container_image_with_retry(
    artifacts_client: ArtifactsClient,
    image_id: str,
):
    """Delete a container image with retry mechanism"""
    try:
        artifacts_client.delete_container_image(image_id=image_id)
    except Exception as e:
        print(f"\nError deleting container image {image_id}: {e}")
        return False
