from u import *
from exp import *
from env import *
import math
class Platoon(Entity):
    pass

class GridEnv(Env):
    def def_sumo(self):
        c = self.c

        types = [E('vType', id='human', **IDM, **LC2013), E('vType', id='rl', **IDM, **LC2013), E('vType', id='generic', **IDM, **LC2013)]
        default_flows = lambda flow_id, route_id, flow_rate: [E('flow', **params) for params in [
            FLOW(f'{flow_id}', type='generic', route=route_id, departSpeed=c.depart_speed, vehsPerHour=flow_rate),
        ] if params.get('vehsPerHour')]
        # Add flow for poisson distribution
        # https://sumo.dlr.de/docs/Simulation/Randomness.html
        poisson_flows = lambda flow_id, route_id, period: [E('flow', **params) for params in [
            FLOW(f'{flow_id}', type='generic', route=route_id, departSpeed=c.depart_speed, period=period),
        ]]

        builder = NetBuilder()
        xys = np.array(np.ones((c.n_rows + 2, c.n_cols + 2)).nonzero()).T * c.length
        nodes = builder.add_nodes(
            [Namespace(x=x, y=y, type='priority') for y, x in xys]
        ).reshape(c.n_rows + 2, c.n_cols + 2)

        tl = c.setdefault('tl', False)
        if tl:
            c.av_frac = 0
            c.pop('av_range', None)
            c.speed_mode = SPEED_MODE.all_checks
        # Store route by direction for route distribution
        routes_by_dir = {direction: [] for direction in c.directions}
        flows = []
        c.setdefaults(flow_rate_h=c.flow_rate, flow_rate_v=c.flow_rate)
        c.log(f'Horizontal flow rate: {c.flow_rate_h}, vertical flow rate: {c.flow_rate_v}')
        priority = ['left', 'right'] if c.get('priority', 'vertical') == 'horizontal' else ['up', 'down']
        for direction in c.directions:
            chains = nodes if direction in ['left', 'right'] else nodes.T
            chains = chains if direction in ['up', 'right'] else np.fliplr(chains)
            flow_rate = c.flow_rate_h if direction in ['left', 'right'] else c.flow_rate_v

            edge_attrs = dict(priority=int(direction in priority))
            if c.get('set_edge_speed', True):
                edge_attrs['speed'] = c.max_speed

            for i, chain in enumerate(chains[1:-1]):
                route_id, flow_id = f'r_{direction}_{i}', f'f_{direction}_{i}'
                _, _, route = builder.chain(chain, route_id=route_id, edge_attrs=edge_attrs)
                routes_by_dir[direction].append(route)
                if c.use_poisson: 
                    period = f"exp({3600 / flow_rate: .2f})"
                    flows.extend(poisson_flows(flow_id, route_id, period))
                else:
                    flows.extend(default_flows(flow_id, route_id, flow_rate))
        # Add turn routes if enabled
        if c.chain_lr:
            connections, turn_routes = builder.chain_leftright(builder.edges.values(), edge_attrs=edge_attrs)
            # for i in range(len(turn_routesroutes)):
            #     flow_id = f'f_turn_{i}'
            #     flows.extend(default_flows(flow_id, turn_routes[i].id, flow_rate))
            

            for route in turn_routes:
                edge_ids = route.edges.split(' ')
                first_edge = edge_ids[0]

                for direction in c.directions:
                    if first_edge == routes_by_dir[direction][0].edges.split(' ')[0]:
                        routes_by_dir[direction].append(route)
                        #print(f'Added turn route {route.id} to direction {direction}')
                        break
            for direction, routes in routes_by_dir.items():
                if not routes:
                    continue
                # Create route distribution
                dist_id = f'dist_{direction}'
                route_dist = E('routeDistribution', id=dist_id)
                prob = 1.0 / len(routes)
                for route in routes:
                    route_dist.append(E('route', refId=route.id, probability=str(prob)))
                builder.additional.append(route_dist)
                for flow in flows:
                    if direction in flow.id:
                        flow.route = dist_id
                
        tls = []
        if tl:
            tl = 1000000 if tl == 'MaxPressure' else tl
            tl_h, tl_v = tl if isinstance(tl, tuple) else (tl, tl)
            tl_offset = c.get('tl_offset', 'auto')
            yellow = c.get('yellow', 0.5)
            if tl_offset == 'auto':
                offsets = c.length * (np.arange(c.n_rows).reshape(-1, 1) + np.arange(c.n_cols).reshape(1, -1)) / 10
            elif tl_offset == 'same':
                offsets = np.zeros(c.n_rows).reshape(-1, 1) + np.zeros(c.n_cols).reshape(1, -1)
            for node, offset in zip(nodes[1:-1, 1:-1].reshape(-1), offsets.reshape(-1)):
                node.type = 'traffic_light'
                phase_multiple = len(c.directions) // 2
                # Add traffic light logic for 4-way intersection with left-right turns
                if c.chain_lr and c.directions == ['up', 'right', 'down', 'left']:
                    tls.append(E('tlLogic',
                    # Phase 1
                    E('phase', duration=tl_v, state='gGGrrrgGGrrr'),
                    *lif(yellow, E('phase', duration=yellow, state='yyyrrryyyrrr')),
                    E('phase', duration=3, state='rrrrrrrrrrrr'),
                    # Phase 2
                    E('phase', duration=tl_h, state='rrrgGGrrrgGG' ),
                    *lif(yellow, E('phase', duration=yellow, state='rrryyyrrryyy')),
                    E('phase', duration=3, state='rrrrrrrrrrrr'),
                id=node.id, offset=offset, type='static', programID='1'))
                else:
                    tls.append(E('tlLogic',
                        E('phase', duration=tl_v, state='Gr' * phase_multiple),
                        *lif(yellow, E('phase', duration=yellow, state='yr' * phase_multiple)),
                        E('phase', duration=tl_h, state='rG' * phase_multiple),
                        *lif(yellow, E('phase', duration=yellow, state='ry' * phase_multiple)),
                    id=node.id, offset=offset, type='static', programID='1'))

        nodes, edges, connections, routes = builder.build()
        additional = E('additional', *types, *routes, *flows, *tls)
        return super().def_sumo(nodes, edges, connections, additional)

    def build_platoon(self):
        ts = self.ts
        rl_type = ts.types.rl
        for route in ts.routes:
            vehs = []
            route_offset = 0
            for edge in route.edges:
                for veh in edge.vehicles:
                    veh.route_position = route_offset + veh.laneposition
                    vehs.append(veh)
                route_offset += edge.length

            rl_mask = np.array([veh.type is rl_type for veh in vehs])
            if len(rl_mask) > 1 and c.get('merge_consecutive_avs'):
                rl_mask[1:] = rl_mask[1:] & ~rl_mask[:-1]
            rl_idxs, = rl_mask.nonzero()
            split_idxs = 1 + rl_idxs

            prev = None
            # Assign platoons to vehicles
            for i, vehs_i in enumerate(np.split(vehs, split_idxs)):
                if not len(vehs_i):
                    continue # Last vehicle is RL, so the last split is empty
                platoon = Platoon(id=f'{route}.platoon_{i}', route=route,
                    vehs=vehs_i, head=vehs_i[-1], tail=vehs_i[0], prev=prev
                )
                if prev is not None:
                    prev.next = platoon
                prev = platoon
                for veh in vehs_i:
                    veh.platoon = platoon
            if prev is not None:
                prev.next = None

    def reset(self):
        c = self.c
        self.mp_tlast = 0
        while not self.reset_sumo():
            pass
        ret = super().init_env()
        return ret

    def step(self, action=[]):
        c = self.c
        ts = self.ts
        max_dist = c.max_dist
        max_speed = c.max_speed

        rl_type = ts.types.rl
        
        prev_rls = sorted(rl_type.vehicles, key=lambda x: x.id)
        # Store vehicle before action
        vehicle_prev = {}
        for veh in ts.vehicles:
            vehicle_prev[veh.id] = Namespace(position=veh.position, 
                                             route_position=veh.route_position, 
                                             speed=veh.speed, 
                                             route=veh.route,
                                             lane=veh.lane,
                                             type=veh.type,
                                             laneposition=veh.laneposition)
            
        for veh, act in zip(prev_rls, action):
            if c.act_type == 'accel':
                level = (np.clip(act, c.low, 1) - c.low) / (1 - c.low)
                ts.accel(veh, (level * 2 - 1) * (c.max_accel if level > 0.5 else c.max_decel))
            else:
                if c.n_actions == 5:
                    accel_map = [-4.5, -c.max_decel, 0, c.max_accel, 2.6]
                    ts.accel(veh, accel_map[act])
                else:
                    level = act / (c.n_actions - 1)
                    ts.accel(veh, (level * 2 - 1) * (c.max_accel if level > 0.5 else c.max_decel))

        if c.tl == 'MaxPressure':
            self.mp_tlast += c.sim_step
            tmin = c.get('mp_tmin', 0)
            if self.mp_tlast >= tmin:
                for tl in ts.traffic_lights:
                    if ts.get_program(tl) == 'off':
                        break
                    jun = tl.junction
                    pressures = [len(p.vehicles) - len(n.vehicles) for p, n in zip(jun.prev_lanes, jun.next_lanes)]

                    total_pressures = []
                    for phase in (ph for ph in tl.phases if 'y' not in ph.state):
                        total_pressures.append(sum(p for p, s in zip(pressures, phase.state) if s == 'G'))

                    ts.set_phase(tl, np.argmax(total_pressures))
                self.mp_tlast = 0

        super().step()
        self.build_platoon()

        obs = {}

        veh_default_close = Namespace(speed=max_speed, route_position=np.inf)
        veh_default_far = Namespace(speed=0, route_position=-np.inf)
        vehs_default = lambda: [veh_default_close] + [veh_default_far] * 2 * c.obs_next_cross_platoons
        for veh in rl_type.vehicles:
            route, lane, platoon = veh.route, veh.lane, veh.platoon
            junction = lane.next_junction

            head, tail = veh, platoon.tail
            route_vehs = [(route, [head, *lif(c.obs_tail, tail)])]

            if junction is ts.sentinel_junction:
                route_vehs.extend([(None, vehs_default())] * (len(c.directions) - 1))
            else:
                # Group crossing lane by direction
                max_cross_directions = len(c.directions) - 1
                lanes_by_direction = {}

                for jun_lane in lane.next_junction_lanes:
                    jun_lane_route = nexti(jun_lane.from_routes)
                    lane_direction = None
                    for direction in c.directions:
                        if direction in jun_lane_route.id:
                            lane_direction = direction
                            break
                    
                    # Keep the lane for each direction
                    if lane_direction and lane_direction not in lanes_by_direction:
                        lanes_by_direction[lane_direction] = jun_lane
                        if len(lanes_by_direction) >= max_cross_directions:
                            break
                for direction in c.directions:
                    if direction in lanes_by_direction:          
                    # Defaults for jun_lane
                        jun_headtails = vehs_default()
                        jun_lane = lanes_by_direction[direction]
                        jun_lane_route = nexti(jun_lane.from_routes)
                        jun_veh, _ = jun_lane.prev_vehicle(0, route=jun_lane_route)
                        jun_veh = jun_veh if jun_veh and jun_veh.lane.next_junction is junction else None

                        if jun_veh:
                            if jun_veh.type is rl_type:
                                # If jun_veh is RL or jun_veh is human and there's no RL vehicle in front of it
                                jun_headtails[1: 3] = jun_veh, jun_veh.platoon.tail
                                platoon = jun_veh.platoon.prev
                                for i in 1 + 2 * np.arange(1, c.obs_next_cross_platoons):
                                    if platoon is None: break
                                    jun_headtails[i: i + 2] = platoon.head, platoon.tail
                                    platoon = platoon.prev
                            else:
                                # If jun_veh is a human vehicle behind some RL vehicle (in another lane)
                                jun_headtails[0] = jun_veh.platoon.tail
                                next_cross_platoon = jun_veh.platoon.prev
                                if next_cross_platoon:
                                    jun_headtails[1: 3] = next_cross_platoon.head, next_cross_platoon.tail
                                    platoon = next_cross_platoon.prev
                                    for i in 1 + 2 * np.arange(1, c.obs_next_cross_platoons):
                                        if platoon is None: break
                                        jun_headtails[i: i + 2] = platoon.head, platoon.tail
                                        platoon = platoon.prev
                        route_vehs.append((jun_lane_route, jun_headtails))
            # Build observation
            ego_pos = np.array(veh.position)
            dist_features, speed_features, turn_features, veh_dist_features = [], [], [], []
            for route, vehs in route_vehs:
                j_pos = junction.route_position[route]                
                for v in vehs:
                    if not math.isinf(v.route_position):
                        v_pos = np.array(v.position)
                        # Distance feature
                        dist = np.linalg.norm(v_pos - ego_pos) / max_dist
                        veh_dist_features.extend([np.clip(dist, 0, 1)])
                    else:
                        veh_dist_features.extend([1])
                    # Turn feature
                    if math.isinf(v.route_position):
                        turn_features.extend([0.5])
                        continue
                    if 'left' in v.route.id:
                        turn_features.extend([0])
                    elif 'right' in v.route.id:
                        turn_features.extend([1])
                    else:
                        turn_features.extend([0.5])    
                dist_features.extend([0 if j_pos == v.route_position else (j_pos - v.route_position) / max_dist for v in vehs])
                speed_features.extend([v.speed / max_speed for v in vehs])
            if c.chain_lr:
                obs[veh.id] = np.clip([*dist_features, *speed_features, *turn_features, *veh_dist_features], 0, 1).astype(np.float32) * (1 - c.low) + c.low
            else: 
                obs[veh.id] = np.clip([*dist_features, *speed_features], 0, 1).astype(np.float32) * (1 - c.low) + c.low 
        sort_id = lambda d: [v for k, v in sorted(d.items())]
        ids = sorted(obs)
        obs = arrayf(sort_id(obs)).reshape(-1, c._n_obs)
        
        
        # If using PPO
        if c.use_critic:
            # Cumulative collision count from rollout_info (already tracked per step)
            collision_count = sum(self.rollout_info['collisions'])
            
            # Progressive penalty: each subsequent collision gets higher penalty
            collision_penalty_multiplier = c.get('collision_penalty_growth', 1.0)
            current_collision_coef = c.collision_coef * (collision_penalty_multiplier ** collision_count)
            # Make reward becomes global reward
            global_reward = len(ts.new_arrived) - current_collision_coef * len(ts.new_collided)
            # Added new individual reward for each RL vehicle
            reward = {}
            penalty_rls = []
            close_rls = {}
            if ts.new_collided:
                penalty_rls = self.check_collided(ts.new_collided, rl_type.vehicles, vehicle_prev)
            if ts.new_arrived:
                close_rls = self.check_arrived(ts.new_arrived, ts.new_collided, prev_rls, vehicle_prev)
            for veh in prev_rls:
                in_reward = 0
                if ts.new_arrived or ts.new_collided:            
                    if veh in ts.new_arrived:
                        in_reward += 2 # Increase reward for arrived vehicles
                    elif close_rls:
                        if veh.id in close_rls:
                            in_reward += close_rls[veh.id]
                    if veh in ts.new_collided:
                        in_reward -= current_collision_coef
                    elif penalty_rls:
                        if veh.id in penalty_rls:
                            in_reward -= current_collision_coef / len(penalty_rls)
                # Add the global reward portion for each RL vehicle
                in_reward += global_reward / len(prev_rls) if len(prev_rls) else 0 
                reward[veh.id] = in_reward
            # Sort and convert to arrays
            reward = arrayf(sort_id(reward))
            return Namespace(obs=obs, id=ids, global_reward=global_reward, reward=reward)
        else:
            reward = len(ts.new_arrived) - c.collision_coef * len(ts.new_collided)
            return Namespace(obs=obs, id=ids, reward=reward)

    def append_step_info(self):
        super().append_step_info()
        self.rollout_info.append(n_veh_network=len(self.ts.vehicles))

    # Penalty for RL vehicles close to collided vehicles in the same lane or at the junction
    def check_collided(self, collided_vehs, rl_vehs, vehicle_prev, c_dist_threshold=20):
        ts = self.ts 
        penalty_rls = []
        for c_veh in collided_vehs:
            c_veh = vehicle_prev[c_veh.id]
            c_pos = np.array(c_veh.position)
            for rl_veh in rl_vehs:
                rl_pos = np.array(rl_veh.position)
                dist = np.linalg.norm(c_pos - rl_pos)
                # Check if RL vehicle is ahead and close to the collided vehicle and in the same lane or at the junction
                if dist < c_dist_threshold and ((rl_veh.lane == c_veh.lane and rl_veh.laneposition < c_veh.laneposition) or ':' in c_veh.lane.id):                    
                    penalty_rls.append(rl_veh.id)
        return penalty_rls
    
    # Reward for RL vehicles close to arrived vehicles in the same lane
    def check_arrived(self, arrived_vehs, collided_vehs, prev_rls, vehicle_prev, a_dist_threshold=60):
        ts = self.ts
        close_rls = {}
        for a_vehs in arrived_vehs: 
            if a_vehs not in prev_rls:
                a_vehs = vehicle_prev[a_vehs.id]
                # Find a RL vehicle near the arrived vehicle
                for rls in prev_rls:
                    if rls in arrived_vehs or rls in collided_vehs:
                        continue
                    if rls.lane != a_vehs.lane:
                        continue
                    distance = np.linalg.norm(np.array(rls.position) - np.array(a_vehs.position))
                    if distance < a_dist_threshold:
                        close_rls[rls.id] = distance / a_dist_threshold
        return close_rls
    @property
    def stats(self):
        c = self.c
        info = self.rollout_info[1 + c.warmup_steps + c.skip_stat_steps:]
        mean = lambda L: np.mean(L) if len(L) else np.nan
        stats = {**super().stats, **dif('length_range' in c, length=c.length), **dif('av_range' in c, av_frac=c.av_frac)}
        stats['backlog_step'] = mean(info['backlog'])
        stats['n_total_veh_step'] = mean(info['n_veh_network']) + stats['backlog_step']
        stats['flow_horizontal'] = c.flow_rate_h
        stats['flow_vertical'] = c.flow_rate_v
        return stats

class GridExp(Main):
    def create_env(c):
        return NormEnv(c, GridEnv(c))

    @property
    def observation_space(c):
        low = np.full(c._n_obs, c.low)
        return Box(low, np.ones_like(low))

    @property
    def action_space(c):
        if c.act_type == 'accel':
            return Box(low=c.low, high=1, shape=(1,), dtype=np.float32)
        else:
            return Discrete(c.n_actions)
    # Comment on_rollout_end since not used
    # def on_rollout_end(c, rollout, stats, ii=None, n_ii=None):
    #     log = c.get_log_ii(ii, n_ii)
    #     step_obs_ = rollout.obs
    #     step_obs = step_obs_[:-1]
        
    #     ret, adv = calc_adv(rollout.reward, c.gamma, rollout.value)

    #     n_veh = np.array([len(o) for o in step_obs])
    #     step_ret = [[r] * nv for r, nv in zip(ret, n_veh)]
    #     rollout.update(obs=step_obs, ret=step_ret, adv=adv)

    #     step_id_ = rollout.pop('id')
    #     id = np.concatenate(step_id_[:-1])
    #     id_unique = np.unique(id)

    #     reward = np.array(rollout.pop('reward'))
    #     raw_reward = np.array(rollout.pop('raw_reward'))

    #     log(**stats)
    #     log(raw_reward_mean=raw_reward.mean(), raw_reward_sum=raw_reward.sum())
    #     log(reward_mean=reward.mean(), reward_sum=reward.sum())
    #     log(n_veh_step_mean=n_veh.mean(), n_veh_step_sum=n_veh.sum(), n_veh_unique=len(id_unique))
    #     return rollout

if __name__ == '__main__':
    c = GridExp.from_args(globals(), locals()) # Initialize configuration 
    c.setdefaults(
        n_steps=300,
        step_save=5,

        depart_speed=0,
        max_speed=13,
        max_dist=100,
        max_accel=1.5,
        max_decel=3.5,
        sim_step=0.5,
        generic_type=True,
        n_actions=3,

        adv_norm=False,
        batch_concat=True,

        render=False,

        warmup_steps=100,
        horizon=2000,
        #directions=['up', 'right'],
        directions='4way',
        av_frac=0.15,
        flow_rate=700,
        length=100,
        n_rows=2,
        n_cols=1,
        speed_mode=SPEED_MODE.obey_safe_speed,

        act_type='accel_discrete',
        low=-1,

        alg=PG,
        n_gds=1,
        lr=1e-3,
        gamma=0.99,
        collision_coef=5, # If there's a collision, it always involves an even number of vehicles

        norm_reward=True,
        center_reward=True,
        opt='RMSprop',

        obs_tail=True,
        obs_next_cross_platoons=1,
    )
    if c.directions == '4way':
        c.directions = ['up', 'right', 'down', 'left']
    if c.chain_lr:
        # Added turn feature to observation  
        c._n_obs = 4 * (1 + c.obs_tail + (1 + 2 * c.obs_next_cross_platoons) * (len(c.directions) - 1))
    else:
        c._n_obs = 2 * (1 + c.obs_tail + (1 + 2 * c.obs_next_cross_platoons) * (len(c.directions) - 1)) 
    #assert c.alg == PG, 'Not supporting value functions yet'
    c.run()