from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Lock

import oci
from oci.object_storage import ObjectStorageClient
from oci.object_storage.models import (
    ObjectVersionSummary,
)
from tqdm import tqdm

from bucket.sdk import (
    abort_multipart_upload_with_retry,
    delete_bucket_with_retry,
    delete_object_with_retry,
    delete_par_with_retry,
    list_multipart_uploads,
    list_object_versions,
    list_preauthenticated_requests,
    verify_bucket_exists,
)


def delete_object_worker(
    object_storage_client: ObjectStorageClient,
    namespace: str,
    bucket_name: str,
    queue: Queue[ObjectVersionSummary],
    progress_lock: Lock,
    obj_pbar,
):
    """Worker function to delete objects from the queue"""
    while True:
        try:
            item: ObjectVersionSummary = queue.get_nowait()
            object_name = item.name
            version_id = item.version_id

            try:
                delete_object_with_retry(
                    object_storage_client,
                    namespace,
                    bucket_name,
                    object_name,
                    version_id,
                )
                with progress_lock:
                    percentage = (obj_pbar.n + 1) / obj_pbar.total * 100
                    obj_pbar.set_postfix_str(
                        f"[{percentage:.1f}%] Deleted: {object_name}"
                    )
                    obj_pbar.update(1)
            except Exception as e:
                print(f"\nError deleting {object_name} | Version ID: {version_id}: {e}")
                with progress_lock:
                    obj_pbar.update(1)
            finally:
                queue.task_done()

        except Empty:
            break


def clean_object_versions_by_prefix(
    object_storage_client: ObjectStorageClient,
    bucket_name: str,
    namespace: str,
    prefix: str,
    num_workers: int = 1,
) -> bool:
    """Delete object versions whose names start with the given prefix."""
    print(
        f"\nDeleting object versions in bucket '{bucket_name}' with prefix '{prefix}'"
    )

    if not verify_bucket_exists(object_storage_client, namespace, bucket_name):
        return False

    objects: list[ObjectVersionSummary] = list_object_versions(
        object_storage_client, bucket_name, namespace, prefix
    )

    if not objects:
        print(
            f"No object versions found with prefix '{prefix}' in bucket '{bucket_name}'"
        )
        return True

    object_queue: Queue[ObjectVersionSummary] = Queue()
    progress_lock = Lock()
    print(f"Running with {num_workers} workers")

    with tqdm(
        total=len(objects),
        desc=f"Deleting object versions in {bucket_name}",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        leave=False,
    ) as obj_pbar:
        for item in objects:
            object_queue.put(item)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    delete_object_worker,
                    object_storage_client,
                    namespace,
                    bucket_name,
                    object_queue,
                    progress_lock,
                    obj_pbar,
                )
                for _ in range(num_workers)
            ]

            for future in futures:
                future.result()

            object_queue.join()

    return True


def clean_up_bucket(
    object_storage_client: ObjectStorageClient,
    bucket_name: str,
    namespace: str,
    bucket_pbar=None,
    delete_bucket=True,
    num_workers=1,
):
    """Delete all objects from a bucket and then delete the bucket itself if delete_bucket is True"""
    bucket_desc = f"Cleaning bucket: {bucket_name}"
    if bucket_pbar:
        bucket_pbar.set_description(bucket_desc)
    else:
        print(f"\n{bucket_desc}")

    # Verify bucket exists
    if not verify_bucket_exists(object_storage_client, namespace, bucket_name):
        if bucket_pbar:
            bucket_pbar.update(1)
        return False

    # Get a list of object versions
    objects: list[ObjectVersionSummary] = list_object_versions(
        object_storage_client, bucket_name, namespace
    )

    if not objects:
        print(f"No objects found in bucket '{bucket_name}'")
    else:
        # Create a queue and progress bar for object deletion
        object_queue: Queue[ObjectVersionSummary] = Queue()
        progress_lock = Lock()
        print(f"Running with {num_workers} workers")

        # Create a progress bar with a percentage format
        with tqdm(
            total=len(objects),
            desc=f"Deleting objects in {bucket_name}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            leave=False,
        ) as obj_pbar:
            # Fill the queue with objects
            for item in objects:
                object_queue.put(item)

            # Create and start worker threads
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit worker tasks
                futures = [
                    executor.submit(
                        delete_object_worker,
                        object_storage_client,
                        namespace,
                        bucket_name,
                        object_queue,
                        progress_lock,
                        obj_pbar,
                    )
                    for _ in range(num_workers)
                ]

                # Wait for all tasks to complete
                for future in futures:
                    future.result()

                # Wait for the queue to be empty
                object_queue.join()

    # Delete all preauthenticated requests
    pars = list_preauthenticated_requests(object_storage_client, namespace, bucket_name)
    if pars:
        with tqdm(
            total=len(pars),
            desc=f"Deleting pre-authenticated requests in {bucket_name}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            leave=False,
        ) as par_pbar:
            for par in pars:
                try:
                    delete_par_with_retry(
                        object_storage_client, namespace, bucket_name, par.id
                    )
                    percentage = (par_pbar.n + 1) / par_pbar.total * 100
                    par_pbar.set_postfix_str(
                        f"[{percentage:.1f}%] Deleted PAR: {par.id}"
                    )
                except Exception as e:
                    print(f"\nError deleting PAR {par.id}: {e}")

                par_pbar.update(1)
    else:
        print(f"No pre-authenticated requests found in bucket '{bucket_name}'")

    # Abort all multipart uploads
    multipart_uploads = list_multipart_uploads(
        object_storage_client, namespace, bucket_name
    )
    if multipart_uploads:
        with tqdm(
            total=len(multipart_uploads),
            desc=f"Aborting multipart uploads in {bucket_name}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            leave=False,
        ) as upload_pbar:
            for upload in multipart_uploads:
                try:
                    abort_multipart_upload_with_retry(
                        object_storage_client,
                        namespace,
                        bucket_name,
                        upload.object,
                        upload.upload_id,
                    )
                    percentage = (upload_pbar.n + 1) / upload_pbar.total * 100
                    upload_pbar.set_postfix_str(
                        f"[{percentage:.1f}%] Aborted upload: {upload.object}"
                    )
                except Exception as e:
                    print(f"\nError aborting multipart upload for {upload.object}: {e}")

                upload_pbar.update(1)
    else:
        print(f"No multipart uploads found in bucket '{bucket_name}'")

    # After all objects, PARs, and multipart uploads are deleted, delete the bucket if requested
    success = True
    if delete_bucket:
        success = delete_bucket_with_retry(
            object_storage_client, namespace, bucket_name
        )
    else:
        print(f"\nSkipping bucket deletion for '{bucket_name}' as requested")

    if bucket_pbar:
        bucket_pbar.update(1)

    return success


def clean_up_buckets_from_file(
    oci_profile: str,
    bucket_file: str,
    namespace: str,
    delete_bucket: bool = True,
    num_workers: int = 1,
):
    """Clean up multiple buckets listed in a file"""
    # Load OCI config from specified profile
    config = oci.config.from_file(profile_name=oci_profile)
    object_storage_client: ObjectStorageClient = oci.object_storage.ObjectStorageClient(
        config
    )

    # Read bucket names from a file
    try:
        with open(bucket_file, "r") as f:
            buckets = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Bucket list file '{bucket_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading bucket list file: {e}")
        return

    if not buckets:
        print("No buckets found in the file.")
        return

    print(f"\nStarting cleanup of {len(buckets)} buckets...")

    # Create a progress bar for buckets
    with tqdm(total=len(buckets), desc="Overall progress", position=0) as bucket_pbar:
        for bucket_name in buckets:
            clean_up_bucket(
                object_storage_client,
                bucket_name,
                namespace,
                bucket_pbar,
                delete_bucket,
                num_workers,
            )
