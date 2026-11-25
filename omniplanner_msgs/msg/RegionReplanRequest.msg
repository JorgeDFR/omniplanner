# Request a repair after the executor has stopped this plan.
string robot_id
string plan_id
string reason
# Corresponding endpoints of edges unavailable to the replacement plan.
string[] blocked_sources
string[] blocked_targets
# Confirmed successful prefix reported by executors with indexed results.
uint32 completed_actions
# Graph waypoints reached during an interrupted or failed navigation action.
string[] reached_places
