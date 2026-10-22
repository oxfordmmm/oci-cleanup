
from oci.log_analytics import LogAnalyticsClient
from oci.log_analytics.models import LogAnalyticsEntitySummary
from tenacity import retry, stop_after_attempt, wait_fixed


def list_log_analytics_entities(
    log_analytics_client: LogAnalyticsClient, compartment_id: str, namespace: str
) -> list[LogAnalyticsEntitySummary]:
    """List all log analytics entities in a compartment with pagination"""
    try:
        all_entities: list[LogAnalyticsEntitySummary] = []
        next_page = None

        while True:
            response = log_analytics_client.list_log_analytics_entities(
                namespace_name=namespace,
                compartment_id=compartment_id,
                page=next_page,
                limit=1000,
            )

            if response.data.items:
                all_entities.extend(response.data.items)

            # Check if there are more items
            next_page = response.headers.get("opc-next-page")
            if not next_page:
                break

        return all_entities
    except Exception as e:
        print(f"Error listing log analytics entities: {e}")
        return []


@retry(stop=stop_after_attempt(4), wait=wait_fixed(10))
def delete_log_analytics_entity_with_retry(
    log_analytics_client: LogAnalyticsClient, namespace: str, entity_id: str
):
    """Delete a log analytics entity with retry mechanism"""
    try:
        log_analytics_client.delete_log_analytics_entity(
            namespace_name=namespace,
            log_analytics_entity_id=entity_id,
        )
        return True
    except Exception as e:
        print(f"\nError deleting log analytics entity {entity_id}: {e}")
        return False
