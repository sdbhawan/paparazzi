#ifndef WAYPOINTS_LIST_H
#define WAYPOINTS_LIST_H

#include "modules/nav/waypoints.h"
#include "pprzlink/messages.h"

#ifdef __cplusplus
extern "C" {
#endif

void send_wp_list(struct transport_tx *trans, struct link_device *dev);
void waypoints_list_init(void);   // <-- add this!

#ifdef __cplusplus
}
#endif

#endif /* WAYPOINTS_LIST_H */

