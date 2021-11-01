#!/usr/bin/env python3
"""The CarRacing-v0 environment from Gym, adapted not to use OpenGL"""
import math
import sys

import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import colorize, seeding, EzPickle

###################
# car_dynamics.py #
###################

# Top-down car dynamics simulation.
#
# Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
# This simulation is a bit more detailed, with wheels rotation.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

SIZE = 0.02
ENGINE_POWER = 100000000*SIZE*SIZE
WHEEL_MOMENT_OF_INERTIA = 4000*SIZE*SIZE
FRICTION_LIMIT = 1000000*SIZE*SIZE     # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [
    (-55, +80), (+55, +80),
    (-55, -82), (+55, -82)
    ]
HULL_POLY1 = [
    (-60, +130), (+60, +130),
    (+60, +110), (-60, +110)
    ]
HULL_POLY2 = [
    (-15, +120), (+15, +120),
    (+20, +20), (-20, 20)
    ]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20)
    ]
HULL_POLY4 = [
    (-50, -120), (+50, -120),
    (+50, -90),  (-50, -90)
    ]
WHEEL_COLOR = (0.0, 0.0, 0.0)
WHEEL_WHITE = (0.3, 0.3, 0.3)
MUD_COLOR = (0.4, 0.4, 0.0)

class Car:
    def __init__(self, world, init_angle, init_x, init_y):
        self.world = world
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            angle = init_angle,
            fixtures = [
                fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY1 ]), density=1.0),
                fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY2 ]), density=1.0),
                fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY3 ]), density=1.0),
                fixtureDef(shape = polygonShape(vertices=[ (x*SIZE,y*SIZE) for x,y in HULL_POLY4 ]), density=1.0)
                ]
            )
        self.hull.color = (0.8, 0.0, 0.0)
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W, +WHEEL_R), (+WHEEL_W, +WHEEL_R),
            (+WHEEL_W, -WHEEL_R), (-WHEEL_W, -WHEEL_R)
            ]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x+wx*SIZE, init_y+wy*SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x*front_k*SIZE,y*front_k*SIZE) for x, y in WHEEL_POLY]),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0)
                    )
            w.wheel_rad = front_k*WHEEL_R*SIZE
            w.color = WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx*SIZE, wy*SIZE),
                localAnchorB=(0,0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180*900*SIZE*SIZE,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
                )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self.hull]
        self.particles = []

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1: diff = 0.1  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir*min(50.0*val, 3.0)

            # Position => friction_limit
            grass = True
            friction_limit = FRICTION_LIMIT*0.6  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(friction_limit, FRICTION_LIMIT*tile.road_friction)
                grass = False

            # Force
            forw = w.GetWorldVector( (0,1) )
            side = w.GetWorldVector( (1,0) )
            v = w.linearVelocity
            vf = forw[0]*v[0] + forw[1]*v[1]  # forward speed
            vs = side[0]*v[0] + side[1]*v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

            # add small coef not to divide by zero
            w.omega += dt*ENGINE_POWER*w.gas/WHEEL_MOMENT_OF_INERTIA/(abs(w.omega)+5.0)
            self.fuel_spent += dt*ENGINE_POWER*w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15    # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE*w.brake
                if abs(val) > abs(w.omega): val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir*val
            w.phase += w.omega*dt

            vr = w.omega*w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr        # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000*SIZE*SIZE
            p_force *= 205000*SIZE*SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0*friction_limit:
                if w.skid_particle and w.skid_particle.grass == grass and len(w.skid_particle.poly) < 30:
                    w.skid_particle.poly.append( (w.position[0], w.position[1]) )
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle( w.skid_start, w.position, grass )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt*f_force*w.wheel_rad/WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter( (
                p_force*side[0] + f_force*forw[0],
                p_force*side[1] + f_force*forw[1]), True )

    def draw(self, viewer, draw_particles=True):
        if draw_particles:
            for p in self.particles:
                viewer.draw_polyline(p.poly, color=p.color, linewidth=5)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=obj.color)
                if "phase" not in obj.__dict__: continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1 > 0 and s2 > 0: continue
                if s1 > 0: c1 = np.sign(c1)
                if s2 > 0: c2 = np.sign(c2)
                white_poly = [
                    (-WHEEL_W*SIZE, +WHEEL_R*c1*SIZE), (+WHEEL_W*SIZE, +WHEEL_R*c1*SIZE),
                    (+WHEEL_W*SIZE, +WHEEL_R*c2*SIZE), (-WHEEL_W*SIZE, +WHEEL_R*c2*SIZE)
                    ]
                viewer.draw_polygon([trans*v for v in white_poly], color=WHEEL_WHITE)

    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass
        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []

#####################################
# Modified version of car_racing.py #
#####################################

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discrete control is reasonable in this environment as well, on/off discretization is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# The reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles visited in the track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# The game is solved when the agent consistently gets 900+ points. The generated track is random every episode.
#
# The episode finishes when all the tiles are visited. The car also can go outside of the PLAYFIELD -  that
# is far off the track, then it will get -100 and die.
#
# Some indicators are shown at the bottom of the window along with the state RGB buffer. From
# left to right: the true speed, four ABS sensors, the steering wheel position and gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's a powerful rear-wheel drive car -  don't press the accelerator and turn at the
# same time.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

STATE_W = 96
STATE_H = 96
RENDER_UPSCALE = 6

SCALE = 6.0             # Track scale
TRACK_RAD = 900/SCALE   # Track is heavily morphed circle with this radius
PLAYFIELD = 2000/SCALE  # Game over boundary
FPS = 50                # Frames per second
ZOOM = 2.7              # Camera zoom
ZOOM_FOLLOW = True      # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40/SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0/len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)

class CarRacingSoft(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : FPS
    }

    color_black = np.array([0., 0., 0.])
    color_white = np.array([1., 1., 1.])
    color_red = np.array([1., 0., 0.])
    color_green = np.array([0., 1., 0.])
    color_grass_dark = np.array([0.4, 0.8, 0.4])
    color_grass_light = np.array([0.4, 0.9, 0.4])
    color_abs_light = np.array([0., 0., 1.])
    color_abs_dark = np.array([0.2, 0., 1.])

    def __init__(self, frame_skip, verbose=False):
        EzPickle.__init__(self)

        if frame_skip < 1:
            raise ValueError("The value of frame_skip must be at least 1")

        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(
                shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))

        self.action_space = spaces.Box(np.array([-1, 0, 0], dtype=np.float32),
                                       np.array([+1, +1, +1], dtype=np.float32),
                                       dtype=np.float32)  # steer, gas, brake

        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.float32)
        self.state = np.zeros([STATE_H, STATE_W, 3], dtype=np.float32)
        self.frame_skip = frame_skip

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2*math.pi*c/CHECKPOINTS + self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5*TRACK_RAD
            if c == CHECKPOINTS-1:
                alpha = 2*math.pi*c/CHECKPOINTS
                self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                rad = 1.5*TRACK_RAD
            checkpoints.append((alpha, rad*math.cos(alpha), rad*math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5*TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True: # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5*math.pi:
                 beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi:
                 beta += 2*math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                 beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
            if proj < -0.3:
                 beta += min(TRACK_TURN_RATE, abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append((alpha,prev_beta*0.5 + beta*0.5,x,y))
            if laps > 4:
                 break
            no_freeze -= 1
            if no_freeze == 0:
                 break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x*(track[0][2] - track[-1][2])) +
            np.square(first_perp_y*(track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False]*len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE*0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i-1]
            road1_l = (x1 - TRACK_WIDTH*math.cos(beta1), y1 - TRACK_WIDTH*math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH*math.cos(beta1), y1 + TRACK_WIDTH*math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH*math.cos(beta2), y2 - TRACK_WIDTH*math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH*math.cos(beta2), y2 + TRACK_WIDTH*math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01*(i%3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(( [road1_l, road1_r, road2_r, road2_l], t.color ))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH+BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH+BORDER)*math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH+BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH+BORDER) * math.sin(beta2))
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))
        self.track = track
        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.human_render = False
        self.frames = 0

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print("retry to generate track (normal if there are not many instances of this message)")
        self.car = Car(self.world, *self.track[0][1:4])

        return self.step(None)[0]

    def step(self, action):
        total_reward = 0
        for _ in range(self.frame_skip):
            if action is not None:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])

            self.car.step(1.0/FPS)
            self.world.Step(1.0/FPS, 6*30, 2*30)
            self.t += 1.0/FPS

            step_reward = 0
            done = False
            if action is not None: # First step without action, called from reset()
                self.reward -= 0.1
                # We actually don't want to count fuel spent, we want car to be faster.
                # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
                self.car.fuel_spent = 0.0
                step_reward = self.reward - self.prev_reward
                self.prev_reward = self.reward
                if self.tile_visited_count == len(self.track):
                    done = True
                x, y = self.car.hull.position
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    done = True
                    step_reward = -100

            total_reward += step_reward
            self.frames += 1
            if self.frames > 1000: done = True
            if done or action is None: break

        self._draw()
        return np.copy(self.state), total_reward, done, {}

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()

        self.viewer.imshow((self.state.repeat(RENDER_UPSCALE, axis=0).repeat(RENDER_UPSCALE, axis=1)*255).astype(np.uint8))

    def _draw(self):
        # Simple 2D affine transformation class
        class Transform():
            def __init__(self, *values):
                self.matrix = values if len(values) else [1., 0., 0., 0., 1., 0., 0., 0., 1.]

            @staticmethod
            def translation(x, y):
                return Transform(1.0, 0.0, x,
                                 0.0, 1.0, y,
                                 0.0, 0.0, 1.0)

            @staticmethod
            def scale(x, y):
                return Transform(x,   0.0, 0.0,
                                 0.0, y,   0.0,
                                 0.0, 0.0, 1.0)

            @staticmethod
            def rotation(angle):
                cos, sin = math.cos(angle), math.sin(angle)
                return Transform(cos, -sin, 0.0,
                                 sin, cos, 0.0,
                                 0.0, 0.0, 1.0)

            def apply_and_swap(self, point):
                sa, sb, sc, sd, se, sf, _, _, _ = self.matrix
                x, y = point
                return (x * sd + y * se + sf, x * sa + y * sb + sc)

            def __mul__(self, other):
                sa, sb, sc, sd, se, sf, _, _, _ = self.matrix
                oa, ob, oc, od, oe, of, _, _, _ = other.matrix
                return Transform(sa * oa + sb * od, sa * ob + sb * oe, sa * oc + sb * of + sc,
                                 sd * oa + se * od, sd * ob + se * oe, sd * oc + se * of + sf,
                                 0.0, 0.0, 1.0)

            def __imul__(self, other):
                return self.__mul__(other)

        class Renderer():
            def __init__(self, env):
                self.env = env
            def draw_polygon(self, path, color):
                self.env._fill_polygon(path, self.env.state, color)

        if "t" not in self.__dict__: return  # reset() not called yet

        zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   # Animate zoom first second
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform = Transform.translation(STATE_W/2, STATE_H*3/4)
        self.transform *= Transform.scale(STATE_W/1000, STATE_H/1000)
        self.transform *= Transform.scale(zoom, -zoom)
        self.transform *= Transform.rotation(angle)
        self.transform *= Transform.translation(-scroll_x, -scroll_y)

        # Clear
        self.state[:, :, :] = self.color_black

        # Draw road, car and indicators
        self._render_road(scroll_x, scroll_y, zoom)
        self.car.draw(Renderer(self), False)
        self._render_indicators()


    def _render_road(self, scroll_x, scroll_y, zoom):
        self._fill_polygon([
            (-PLAYFIELD, +PLAYFIELD),
            (+PLAYFIELD, +PLAYFIELD),
            (+PLAYFIELD, -PLAYFIELD),
            (-PLAYFIELD, -PLAYFIELD)], self.state, self.color_grass_dark)
        k = PLAYFIELD/20.0
        mindist = 2000000 / (zoom ** 2)
        for x in range(-20, 20, 2):
            kx = k*x
            dist = (kx - scroll_x) ** 2
            if dist >= mindist: continue
            for y in range(-20, 20, 2):
                ky = k * y
                if dist + (ky - scroll_y) ** 2 >= mindist: continue
                self._fill_polygon([
                    (kx + k, ky + 0),
                    (kx + 0, ky + 0),
                    (kx + 0, ky + k),
                    (kx + k, ky + k)], self.state, self.color_grass_light)
        for poly, color in self.road_poly:
            if (poly[0][0] - scroll_x) ** 2 + (poly[0][1] - scroll_y) ** 2 >= mindist: continue
            self._fill_polygon(poly, self.state, color)

    def _render_indicators(self):
        s = STATE_W/40
        h = STATE_H/40
        self._fill_polygon([(0, STATE_H), (STATE_W, STATE_H), (STATE_W, STATE_H - 5*h), (0, STATE_H - 5*h)], self.state,
                           self.color_black, transform=False)
        def vertical_ind(place, val, color):
            self._fill_polygon([((place+0)*s, STATE_H-h-h*val),
                                ((place+2)*s, STATE_H-h-h*val),
                                ((place+2)*s, STATE_H-h),
                                ((place+0)*s, STATE_H-h)], self.state, color, transform=False)
        def horiz_ind(place, val, color):
            self._fill_polygon([((place+0)*s, STATE_H-4*h),
                                ((place+val)*s, STATE_H-4*h),
                                ((place+val)*s, STATE_H-1.5*h),
                                ((place+0)*s, STATE_H-1.5*h)], self.state, color, transform=False)
        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(1, 0.02*true_speed, self.color_white)
        vertical_ind(4, 0.01*self.car.wheels[0].omega, self.color_abs_light) # ABS sensors
        vertical_ind(6, 0.01*self.car.wheels[1].omega, self.color_abs_light)
        vertical_ind(8, 0.01*self.car.wheels[2].omega, self.color_abs_dark)
        vertical_ind(10,0.01*self.car.wheels[3].omega, self.color_abs_dark)
        horiz_ind(20, -10.0*self.car.wheels[0].joint.angle, self.color_green)
        horiz_ind(30, -0.8*self.car.hull.angularVelocity, self.color_red)

    # Adapted from https://github.com/luispedro/mahotas/blob/master/mahotas/polygon.py
    def _fill_polygon(self, polygon, canvas, color, transform=True):
        '''
        fill_polygon([(y0,x0), (y1,x1),...], canvas, color=1)
        Draw a filled polygon in canvas
        Parameters
        ----------
        polygon : list of pairs
            a list of (y,x) points
        canvas : ndarray
            where to draw, will be modified in place
        color : integer, optional
            which colour to use (default: 1)
        '''
        # algorithm adapted from: http://www.alienryderflex.com/polygon_fill/
        if not len(polygon):
            return

        if transform:
            polygon = [self.transform.apply_and_swap(point) for point in polygon]
        else:
            polygon = [(float(y), float(x)) for x, y in polygon]

        min_y = max(int(min(y for y,x in polygon)), 0)
        if min_y >= canvas.shape[0]: return
        max_y = min(max(int(max(y + 1 for y,x in polygon)), 0), canvas.shape[0])
        if max_y <= 0: return
        if min(x for y,x in polygon) >= canvas.shape[1]: return
        if max(x for y,x in polygon) < 0: return
        for y in range(min_y, max_y):
            nodes = []
            j = -1
            for i,p in enumerate(polygon):
                pj = polygon[j]
                if p[0] < y and pj[0] >= y or pj[0] < y and p[0] >= y:
                    dy = pj[0] - p[0]
                    if dy:
                        nodes.append( (p[1] + (y-p[0])/(pj[0]-p[0])*(pj[1]-p[1])) )
                    elif p[0] == y:
                        nodes.append(p[1])
                j = i
            nodes.sort()
            for n,nn in zip(nodes[::2],nodes[1::2]):
                canvas[y, max(int(n), 0):min(max(int(nn), 0), canvas.shape[1])] = color


#################################
# Environment for NPFL122 class #
#################################

FRAME_SKIPS = range(1, 10)
for frame_skip in FRAME_SKIPS:
    gym.envs.register(
        id="CarRacingSoftFS{}-v0".format(frame_skip),
        entry_point=CarRacingSoft,
        kwargs={"frame_skip": frame_skip},
        reward_threshold=900,
    )

# Allow running the environment and  controlling it with arrows
if __name__=="__main__":
    import pyglet
    import time

    from pyglet.window import key
    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  a[0] = -1.0
        if k == key.RIGHT: a[0] = +1.0
        if k == key.UP:    a[1] = +1.0
        if k == key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT  and a[0] == -1.0: a[0] = 0
        if k == key.RIGHT and a[0] == +1.0: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0
    env = CarRacingSoft(1)
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            env.render()
            time.sleep(0.01)
            if done or restart:
                break
    env.close()
