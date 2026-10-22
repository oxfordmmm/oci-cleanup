from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Lock

import oci
from oci.artifacts import ArtifactsClient
from tqdm import tqdm


def delete_container_image_worker(
    artifacts_client: ArtifactsClient,
    queue: Queue[str],
    progress_lock: Lock,
    image_pbar,
):
    """Worker function to delete container images from the queue"""
    while True:
        try:
            image_id = queue.get_nowait()

            try:
                delete_container_image_with_retry(artifacts_client, image_id)
                with progress_lock:
                    percentage = (image_pbar.n + 1) / image_pbar.total * 100
                    image_pbar.set_postfix_str(
                        f"[{percentage:.1f}%] Deleted: {image_id}"
                    )
                    image_pbar.update(1)
            except Exception as e:
                print(f"\nError deleting image ID {image_id}: {e}")
                with progress_lock:
                    image_pbar.update(1)
            finally:
                queue.task_done()

        except Empty:
            break


def clean_up_container_images_from_file(
    oci_profile: str,
    image_file: str,
    num_workers: int = 1,
):
    """Clean up container images from image OCIDs listed in a file"""
    config = oci.config.from_file(profile_name=oci_profile)
    artifacts_client: ArtifactsClient = oci.artifacts.ArtifactsClient(config)

    try:
        with open(image_file, "r") as f:
            image_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Image list file '{image_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading image list file: {e}")
        return

    if not image_ids:
        print("No image IDs found in the file.")
        return

    image_queue: Queue[str] = Queue()
    progress_lock = Lock()
    print(f"\nStarting cleanup of {len(image_ids)} container images...")
    print(f"Running with {num_workers} workers")

    with tqdm(
        total=len(image_ids),
        desc="Deleting container images",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        leave=False,
    ) as image_pbar:
        for image_id in image_ids:
            image_queue.put(image_id)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    delete_container_image_worker,
                    artifacts_client,
                    image_queue,
                    progress_lock,
                    image_pbar,
                )
                for _ in range(num_workers)
            ]

            for future in futures:
                future.result()

            image_queue.join()
