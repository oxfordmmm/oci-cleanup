import oci
from oci.object_storage import ObjectStorageClient
from oci.object_storage.models import (
    MultipartUploadPartSummary,
    ObjectSummary,
    ObjectVersionSummary,
)
from tenacity import retry, stop_after_attempt, wait_fixed


def list_objects(
    object_storage_client: ObjectStorageClient,
    bucket_name: str,
    namespace: str,
    prefix: str | None = None,
) -> list[ObjectSummary]:
    """List all objects in a bucket with pagination (current versions only)."""
    try:
        all_objects: list[ObjectSummary] = []
        start_with: str | None = None

        while True:
            response = object_storage_client.list_objects(
                namespace_name=namespace,
                bucket_name=bucket_name,
                prefix=prefix,
                start=start_with,
                limit=1000,
            )

            if response.data.objects:
                all_objects.extend(response.data.objects)

            # list_objects uses start-based pagination
            start_with = response.data.next_start_with
            if not start_with:
                break

        return all_objects
    except Exception as e:
        print(f"Error listing objects: {e}")
        return []


def list_object_versions(
    object_storage_client: ObjectStorageClient,
    bucket_name: str,
    namespace: str,
    prefix: str | None = None,
) -> list[ObjectVersionSummary]:
    """List all object versions in a bucket with pagination, optionally filtered by prefix"""
    try:
        all_objects: list[ObjectVersionSummary] = []
        next_page = None

        while True:
            response = object_storage_client.list_object_versions(
                namespace_name=namespace,
                bucket_name=bucket_name,
                page=next_page,
                limit=1000,
                prefix=prefix,
            )

            if response.data.items:
                all_objects.extend(response.data.items)

            # Check if there are more items
            next_page = response.headers.get("opc-next-page")
            if not next_page:
                break

        return all_objects
    except Exception as e:
        print(f"Error listing object versions: {e}")
        return []


@retry(stop=stop_after_attempt(4), wait=wait_fixed(10))
def delete_object_with_retry(
    object_storage_client: ObjectStorageClient,
    namespace: str,
    bucket_name: str,
    object_name: str,
    version_id: str,
):
    """Delete an object version with retry mechanism"""
    return object_storage_client.delete_object(
        namespace_name=namespace,
        bucket_name=bucket_name,
        object_name=object_name,
        version_id=version_id,
    )


@retry(stop=stop_after_attempt(4), wait=wait_fixed(10))
def delete_bucket_with_retry(
    object_storage_client: ObjectStorageClient, namespace: str, bucket_name: str
):
    """Delete a bucket with a retry mechanism"""
    try:
        object_storage_client.delete_bucket(
            namespace_name=namespace, bucket_name=bucket_name
        )
        print(f"\nBucket '{bucket_name}' deleted successfully")
        return True
    except oci.exceptions.ServiceError as e:
        if e.code == "BucketNotEmpty":
            print(
                f"\nBucket '{bucket_name}' is not empty. Please ensure all objects are deleted first."
            )
        else:
            print(f"\nError deleting bucket '{bucket_name}': {e}")
        return False


def verify_bucket_exists(
    object_storage_client: ObjectStorageClient, namespace: str, bucket_name: str
):
    """Verify if a bucket exists"""
    try:
        object_storage_client.get_bucket(
            namespace_name=namespace, bucket_name=bucket_name
        )
        return True
    except oci.exceptions.ServiceError as e:
        if e.code == "BucketNotFound":
            print(f"\nBucket '{bucket_name}' does not exist.")
        else:
            print(f"\nError verifying bucket '{bucket_name}': {e}")
        return False


def list_preauthenticated_requests(
    object_storage_client: ObjectStorageClient, namespace: str, bucket_name: str
):
    """List all preauthenticated requests in a bucket with pagination"""
    try:
        all_pars = []
        next_page = None

        while True:
            response = object_storage_client.list_preauthenticated_requests(
                namespace_name=namespace,
                bucket_name=bucket_name,
                page=next_page,
                limit=1000,
            )

            if response.data:
                all_pars.extend(response.data)

            # Check if there are more items
            next_page = response.headers.get("opc-next-page")
            if not next_page:
                break

        return all_pars
    except Exception as e:
        print(
            f"Error listing preauthenticated requests for bucket '{bucket_name}': {e}"
        )
        return []


@retry(stop=stop_after_attempt(4), wait=wait_fixed(10))
def delete_par_with_retry(
    object_storage_client: ObjectStorageClient,
    namespace: str,
    bucket_name: str,
    par_id: str,
):
    """Delete a preauthenticated request with retry mechanism"""
    try:
        object_storage_client.delete_preauthenticated_request(
            namespace_name=namespace, bucket_name=bucket_name, par_id=par_id
        )
        return True
    except Exception as e:
        print(f"\nError deleting preauthenticated request {par_id}: {e}")
        return False


def list_multipart_uploads(
    object_storage_client: ObjectStorageClient, namespace: str, bucket_name: str
) -> list[MultipartUploadPartSummary]:
    """List all multipart uploads in a bucket with pagination"""
    try:
        all_uploads: list[MultipartUploadPartSummary] = []
        next_page = None

        while True:
            response = object_storage_client.list_multipart_uploads(
                namespace_name=namespace,
                bucket_name=bucket_name,
                page=next_page,
                limit=1000,
            )

            if response.data:
                all_uploads.extend(response.data)

            # Check if there are more items
            next_page = response.headers.get("opc-next-page")
            if not next_page:
                break

        return all_uploads
    except Exception as e:
        print(f"Error listing multipart uploads for bucket '{bucket_name}': {e}")
        return []


@retry(stop=stop_after_attempt(4), wait=wait_fixed(10))
def abort_multipart_upload_with_retry(
    object_storage_client: ObjectStorageClient,
    namespace: str,
    bucket_name: str,
    object_name: str,
    upload_id: str,
):
    """Abort a multipart upload with retry mechanism"""
    try:
        object_storage_client.abort_multipart_upload(
            namespace_name=namespace,
            bucket_name=bucket_name,
            object_name=object_name,
            upload_id=upload_id,
        )
        return True
    except Exception as e:
        print(
            f"\nError aborting multipart upload {upload_id} for object {object_name}: {e}"
        )
        return False
