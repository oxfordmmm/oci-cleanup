import click
import oci
from oci import Response
from oci.log_analytics import LogAnalyticsClient
from oci.object_storage import ObjectStorageClient

from bucket.cli_helper import (
    clean_object_versions_by_prefix,
    clean_up_bucket,
    clean_up_buckets_from_file,
)
from bucket.sdk import (
    list_object_versions,
    list_objects,
)
from container.cli_helper import clean_up_container_images_from_file
from log_analytics.cli_helper import clean_log_analytics_entities


@click.group()
def cli():
    """OCI cleanup utilities"""
    pass


@cli.command(name="list-bucket-objects")
@click.option(
    "--oci-profile", required=True, help="OCI profile to use from the config file"
)
@click.option("--bucket-name", required=True, help="Name of the OCI bucket to list")
@click.option("--prefix", default=None, help="Filter objects by name prefix")
@click.option(
    "--versions",
    "should_list_versions",
    is_flag=True,
    help="List/count all object versions (not just current objects)",
)
@click.option(
    "--count",
    "is_count",
    is_flag=True,
    help="Count the number of objects instead of listing them",
)
def list_bucket_objects(
    oci_profile: str,
    bucket_name: str,
    prefix: str | None,
    should_list_versions: bool,
    is_count: bool,
):
    """List or count objects (or all versions) in an OCI bucket."""
    config = oci.config.from_file(profile_name=oci_profile)
    object_storage_client: ObjectStorageClient = oci.object_storage.ObjectStorageClient(
        config
    )
    namespace: str = object_storage_client.get_namespace().data

    if should_list_versions:
        objects = list_object_versions(
            object_storage_client, bucket_name, namespace, prefix
        )
        item_label = "object version(s)"
    else:
        objects = list_objects(object_storage_client, bucket_name, namespace, prefix)
        item_label = "object(s)"

    if is_count:
        print(f"Total number of {item_label} in bucket '{bucket_name}': {len(objects)}")
        return

    print(f"Found {len(objects)} {item_label} in bucket '{bucket_name}':")
    for obj in objects:
        if should_list_versions:
            print(f"  {obj.name}  version={obj.version_id}  size={obj.size}")
        else:
            print(f"  {obj.name}  size={obj.size}")


@cli.command(name="clean-object-versions")
@click.option(
    "--oci-profile", required=True, help="OCI profile to use from the config file"
)
@click.option("--bucket-name", required=True, help="Name of the OCI bucket")
@click.option(
    "--prefix",
    required=True,
    help="Only delete object versions whose names start with this prefix",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Number of worker threads for parallel processing",
)
def clean_object_versions(
    oci_profile: str, bucket_name: str, prefix: str, workers: int
):
    """Delete object versions in a bucket that match a name prefix."""
    config = oci.config.from_file(profile_name=oci_profile)
    object_storage_client: ObjectStorageClient = oci.object_storage.ObjectStorageClient(
        config
    )
    namespace: str = object_storage_client.get_namespace().data

    clean_object_versions_by_prefix(
        object_storage_client,
        bucket_name,
        namespace,
        prefix,
        num_workers=workers,
    )


@cli.command(name="clean-bucket")
@click.option(
    "--oci-profile", required=True, help="OCI profile to use from the config file"
)
@click.option("--bucket-name", help="Single bucket name to clean up")
@click.option(
    "--bucket-file",
    type=click.Path(exists=True),
    help="File containing list of buckets to clean up (one per line)",
)
@click.option(
    "--delete-bucket/--no-delete-bucket",
    default=True,
    help="Delete the bucket after cleaning up its contents",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Number of worker threads for parallel processing",
)
def clean_bucket(
    oci_profile: str,
    bucket_name: str,
    bucket_file: str,
    delete_bucket: bool,
    workers: int,
):
    """Clean up OCI buckets by deleting their contents and optionally the buckets themselves"""
    if not bucket_name and not bucket_file:
        raise click.UsageError(
            "Either --bucket-name or --bucket-file must be specified"
        )

    if bucket_name and bucket_file:
        raise click.UsageError("Cannot specify both --bucket-name and --bucket-file")

    # Initialize OCI client
    config = oci.config.from_file(profile_name=oci_profile)
    object_storage_client: ObjectStorageClient = oci.object_storage.ObjectStorageClient(
        config
    )
    namespace_response: Response = object_storage_client.get_namespace()
    namespace: str = namespace_response.data

    if bucket_file:
        clean_up_buckets_from_file(
            oci_profile, bucket_file, namespace, delete_bucket, workers
        )
    else:
        clean_up_bucket(
            object_storage_client,
            bucket_name,
            namespace,
            delete_bucket=delete_bucket,
            num_workers=workers,
        )


@cli.command(name="clean-logs-analytics")
@click.option(
    "--oci-profile", required=True, help="OCI profile to use from the config file"
)
@click.option(
    "--compartment-id",
    required=True,
    help="Compartment ID containing the log analytics entities",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Number of worker threads for parallel processing",
)
def clean_logs_analytics(oci_profile: str, compartment_id: str, workers: int):
    """Clean up OCI Log Analytics entities in a compartment"""
    # Initialize OCI clients
    config = oci.config.from_file(profile_name=oci_profile)
    log_analytics_client: LogAnalyticsClient = oci.log_analytics.LogAnalyticsClient(
        config
    )
    object_storage_client: ObjectStorageClient = oci.object_storage.ObjectStorageClient(
        config
    )

    # Get namespace using Object Storage client
    try:
        namespace_response: Response = object_storage_client.get_namespace()
        namespace: str = namespace_response.data
    except Exception as e:
        raise click.UsageError(f"Failed to get namespace: {e}")

    clean_log_analytics_entities(
        log_analytics_client, compartment_id, namespace, workers
    )


@cli.command(name="clean-container-image")
@click.option(
    "--oci-profile", required=True, help="OCI profile to use from the config file"
)
@click.option(
    "--image-file",
    required=True,
    type=click.Path(exists=True),
    help="File containing list of container image OCIDs to clean up (one per line)",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Number of worker threads for parallel processing",
)
def clean_container_image(
    oci_profile: str,
    image_file: str,
    workers: int,
):
    """Clean up OCI container images from file input (OCID only)"""
    clean_up_container_images_from_file(
        oci_profile,
        image_file,
        workers,
    )


if __name__ == "__main__":
    cli()
