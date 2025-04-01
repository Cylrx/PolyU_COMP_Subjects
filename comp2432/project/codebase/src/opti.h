#ifndef OPTI_H
#define OPTI_H

#include "input.h"
#include "state.h"

/**
 * @brief Resets the internal state of the opti module. This includes resetting the current temperature, current step, and decay rate.
 * @note This function should be called before starting the optimization process.
 */
void opti_reset();

void opti_rollback(Vector* rejected, Vector* accepted, Tracker* tracker);
void opti_backup(Vector* rejected, Vector* accepted, Tracker* tracker);

/**
 * @brief Improved LJF Greedy algorithm. Move request from rejected to accepted, then resize.
 * @param rejected The vector of rejected requests (will decrease)
 * @param accepted The vector of accepted requests (will increase)
 * @param tracker The segment tree of the current state
 */

void opti_greedy(Vector* rejected, Vector* accepted, Tracker* tracker, bool rand);

/**
 * @brief Delete the request from accepted to rejected, then resize.
 * @param rejected The vector of rejected requests (will increase)
 * @param accepted The vector of accepted requests (will decrease)
 * @param tracker The segment tree of the current state
 * 
 */
void opti_delete(Vector* rejected, Vector* accepted, Tracker* tracker);

/**
 * @return If opti has reached maximum steps
 */
bool opti_running();

/**
 * @brief Calculate the utilization of the accepted requests
 * @param accepted The vector of accepted requests
 * @return The utilization of the accepted requests
 */
double opti_util(Vector* accepted);

/**
 * @brief Accept or reject the new request based on the old utilization
 * @param new_util The new utilization rate
 * @param old_util The old utilization rate
 * @return If the new request should be accepted or not
 */
bool opti_accept(double new_util, double old_util);

/**
 * @brief Iterate the algorithm. Incr current step and decr the current temperature
 */
void opti_iter();

#endif // OPTI_H