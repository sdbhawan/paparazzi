#include "modules/nav/waypoints.h"
#include "modules/nav/waypoints_list.h"
#include "modules/datalink/telemetry.h"
#include "generated/flight_plan.h"

#define NB_WP_MAX 15

#if PERIODIC_TELEMETRY
void send_wp_list(struct transport_tx *trans, struct link_device *dev)
{
    uint8_t n = nb_waypoint;
    if (n > NB_WP_MAX) n = NB_WP_MAX;

    int32_t east[NB_WP_MAX];
    int32_t north[NB_WP_MAX];
    int32_t up[NB_WP_MAX];

    for (uint8_t i = 0; i < n; i++) {
        east[i]  = waypoints[i].enu_i.x;
        north[i] = waypoints[i].enu_i.y;
        up[i]    = waypoints[i].enu_i.z;
    }

    pprz_msg_send_WP_LIST_ENU(
        trans,
        dev,
        AC_ID,
        &n,
        n, east,
        n, north,
        n, up
    );
}
#endif

void waypoints_list_init(void)
{
#if PERIODIC_TELEMETRY
    register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_WP_LIST_ENU, send_wp_list);
#endif
}


// // wrapper so telemetry can call it
// static void send_wp_list_cb(struct transport_tx *trans, struct link_device *dev)
// {
//   send_wp_list(trans, dev);
// }

/** initialize global and local waypoints */
// void waypoints_list_init(void)
// {
//   struct EnuCoor_f wp_tmp_float[NB_WAYPOINT] = WAYPOINTS_ENU;
//   struct LlaCoor_i wp_tmp_lla_i[NB_WAYPOINT] = WAYPOINTS_LLA_WGS84;
//   /* element in array is TRUE if absolute/global waypoint */
//   bool is_global[NB_WAYPOINT] = WAYPOINTS_GLOBAL;
//   uint8_t i = 0;
//   for (i = 0; i < nb_waypoint; i++) {
//     /* clear all flags */
//     waypoints[i].flags = 0;
//     /* init waypoint as global LLA or local ENU */
//     if (is_global[i]) {
//       waypoint_set_global_flag(i);
//       waypoint_set_lla(i, &wp_tmp_lla_i[i]);
//     } else {
//       waypoint_set_enu(i, &wp_tmp_float[i]);
//     }
//   }


// waypoints_list_init() {
//     // Register periodic telemetry
    
// }

// #if PERIODIC_TELEMETRY
//     register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_WP_LIST, send_wp_list);
// #endif



// waypoints_list_init() {
//     // Register periodic telemetry
    
// }




// register telemetry message handler
// TELEMETRY_MSG(WP_LIST_ENU, send_wp_list_cb);
