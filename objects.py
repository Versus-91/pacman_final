import pygame
from pygame.locals import *
import numpy as np
from random import randint
import math

from vector import Vectors as CustomVector

############################################## Constants
cell_side_size = 16
cell_side_midlle = cell_side_size / 2
n_rows = 36
n_columns = 28
window_width = n_columns * cell_side_size
window_height = n_rows * cell_side_size
window_frame_size = (window_width, window_height)

cell = 16  # TILEWIDTH
# cell= 16   #TILEHEIGHT
row = 36  # NROWS
col = 28  # NCOLS
w = col * cell  # SCREENWIDTH
h = row * cell  # SCREENHEIGHT = NROWS*TILEHEIGHT
screen = (w, h)
pi = math.pi

player_images = []
for i in range(1, 5):
    player_images.append(
        pygame.transform.scale(
            pygame.image.load(f"assets/player_images/{i}.png"), (cell * 1.5, cell * 1.5)
        )
    )
pac0 = pygame.transform.scale(
    pygame.image.load(f"assets/player_images/{0}.png"), (cell * 1.5, cell * 1.5)
)

pacman_icon = pygame.transform.scale(
    pygame.image.load(f"assets/Pacman.png"),
    (1.5 * cell_side_size, 1.5 * cell_side_size),
)

blinky_icon = pygame.transform.scale(
    pygame.image.load(f"assets/Blinky.png"),
    (1.5 * cell_side_size, 1.5 * cell_side_size),
)
inky_icon = pygame.transform.scale(
    pygame.image.load(f"assets/Inky.png"), (1.5 * cell_side_size, 1.5 * cell_side_size)
)
pinky_icon = pygame.transform.scale(
    pygame.image.load(f"assets/Pinky.png"), (1.5 * cell_side_size, 1.5 * cell_side_size)
)
clyde_icon = pygame.transform.scale(
    pygame.image.load(f"assets/Clyde.png"), (1.5 * cell_side_size, 1.5 * cell_side_size)
)
dead_icon = pygame.transform.scale(
    pygame.image.load(f"assets/Dead.png"), (1.5 * cell_side_size, 1.5 * cell_side_size)
)
scared_icon = pygame.transform.scale(
    pygame.image.load(f"assets/Scared.png"),
    (1.5 * cell_side_size, 1.5 * cell_side_size),
)


gold_icon = pygame.transform.scale(
    pygame.image.load(f"assets/Gold.png"), (1.5 * cell_side_size, 1.5 * cell_side_size)
)
heart_icon = pygame.transform.scale(
    pygame.image.load(f"assets/Heart.png"), (1.5 * cell_side_size, 1.5 * cell_side_size)
)

# sue
sue_img = pygame.transform.scale(
    pygame.image.load(f"assets/ghost_images/sue.png"),
    (1.5 * cell_side_size, 1.5 * cell_side_size),
)
# funky
funky_img = pygame.transform.scale(
    pygame.image.load(f"assets/ghost_images/funky.png"),
    (1.5 * cell_side_size, 1.5 * cell_side_size),
)


dead = []
for i in range(0, 4):
    dead.append(
        pygame.transform.scale(
            pygame.image.load(f"assets/ghost_images/{i}.png"),
            (1.5 * cell_side_size, 1.5 * cell_side_size),
        )
    )


stop_indicator = 0
up_direction = 1
down_direction = -1
left_direction = 2
right_direction = -2
in_portal = 5

pacman_indicator = 0
objectives_indicator = 1
powerups_indicator = 4
ghosts_indicator = 3
blinky_indicator = 4
pinky_indicator = 15
inky_indicator = 6
clyde_indicator = 7
gold_indicator = 8

sue = 16
funky = 17

scattering_mode = 0
chasing_mode = 1
scared_mode = 2
respawning_mode = 3


############################################## Objects
class MovingObject(object):
    def __init__(self, node):
        self.name = None
        self.directions = {
            up_direction: CustomVector(0, -1),
            down_direction: CustomVector(0, 1),
            left_direction: CustomVector(-1, 0),
            right_direction: CustomVector(1, 0),
            stop_indicator: CustomVector(),
        }
        self.direction = stop_indicator

        self.radius = 10
        self.collision_distance = 5
        self.color = None
        self.visible = True
        self.disablePortal = False
        self.goal = None
        self.moving_method = None
        self.initialNode(node)
        self.image = None
        self.basic_speed = 100
        self.setSpeed(self.basic_speed)

    def setPosition(self):
        self.position = self.node.position.copy()

    def update(self, delta_t):
        self.position += self.directions[self.direction] * self.speed * delta_t

        if self.passedTarget():
            self.node = self.target
            directions = self.allowedDirectionsList()
            direction = self.moving_method(directions)
            if not self.disablePortal:
                if self.node.neighbors[in_portal] is not None:
                    self.node = self.node.neighbors[in_portal]
            self.target = self.setTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.setTarget(self.direction)

            self.setPosition()

    def allowedDirection(self, direction):
        if direction is not stop_indicator:
            if self.name in self.node.access[direction]:
                if self.node.neighbors[direction] is not None:
                    return True
        return False

    def setTarget(self, direction):
        if self.allowedDirection(direction):
            return self.node.neighbors[direction]
        return self.node

    def passedTarget(self):
        if self.target is not None:
            vector_1 = self.target.position - self.node.position
            vector_2 = self.position - self.node.position
            length_1 = vector_1.magnitudeSquared()
            length_2 = vector_2.magnitudeSquared()
            return length_2 >= length_1
        return False

    def turnMove(self):
        self.direction *= -1
        temp_value = self.node
        self.node = self.target
        self.target = temp_value

    def oppositeDirection(self, direction):
        if direction is not stop_indicator:
            if direction == self.direction * -1:
                return True
        return False

    def allowedDirectionsList(self):
        directions = []
        for key in [up_direction, down_direction, left_direction, right_direction]:
            if self.allowedDirection(key):
                if key != self.direction * -1:
                    directions.append(key)
        if len(directions) == 0:
            directions.append(self.direction * -1)
        return directions

    def randomMove(self, directions):
        return directions[randint(0, len(directions) - 1)]

    def toTargetDirection(self, directions):
        distances = []
        for direction in directions:
            vec = (
                self.node.position
                + self.directions[direction] * cell_side_size
                - self.goal
            )
            distances.append(vec.magnitudeSquared())
        index = distances.index(min(distances))
        return directions[index]

    def initialNode(self, node):
        self.node = node
        self.startNode = node
        self.target = node
        self.setPosition()

    def middleOfNodes(self, direction):
        if self.node.neighbors[direction] is not None:
            self.target = self.node.neighbors[direction]
            self.position = (self.node.position + self.target.position) / 2.0

    def reset(self):
        self.initialNode(self.startNode)
        self.direction = stop_indicator
        self.speed = self.basic_speed
        self.visible = True

    def setSpeed(self, speed):
        self.speed = speed

    def render(self, screen):
        if self.visible:
            adjust_position = self.position - CustomVector(
                cell_side_midlle, cell_side_midlle
            )
            screen.blit(self.image, adjust_position.tupleForm())


############################################## Pacman
class Pacman(MovingObject):
    def __init__(self, node):
        MovingObject.__init__(self, node)
        self.name = pacman_indicator
        self.color = "yellow"
        self.direction = left_direction
        self.middleOfNodes(left_direction)
        self.alive = True
        self.image = pacman_icon

    def teleportNode(self, node):
        self.node = node
        self.target = node
        self.setPosition()

    def teleport(self, new_place):
        self.teleportNode(new_place)
        self.direction = right_direction
        self.middleOfNodes(right_direction)

    def reset(self):
        MovingObject.reset(self)
        self.direction = left_direction
        self.middleOfNodes(left_direction)
        self.alive = True

    def die(self):
        self.alive = False
        self.direction = stop_indicator

    def update(self, delta_t,action = None):
        self.position += self.directions[self.direction] * self.speed * delta_t
        direction = self.getInput() if action is None else action
        if self.passedTarget(): 
            self.node = self.target
            if self.node.neighbors[in_portal] is not None:
                self.node = self.node.neighbors[in_portal]
            self.target = self.setTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.setTarget(self.direction)

            if self.target is self.node:
                self.direction = stop_indicator
            self.setPosition()
        else:
            if self.oppositeDirection(direction):
                self.turnMove()

    def getInput(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_UP]:
            return up_direction
        if key_pressed[K_DOWN]:
            return down_direction
        if key_pressed[K_LEFT]:
            return left_direction
        if key_pressed[K_RIGHT]:
            return right_direction
        return stop_indicator

    def collideCheck(self, other_unit):
        distance_squared = (self.position - other_unit.position).magnitudeSquared()
        colider_squared = (self.collision_distance + other_unit.radius) ** 2
        # colider_squared = (self.collision_distance + other_unit.collision_distance)**2
        if distance_squared <= colider_squared:
            return True
        return False

    def collectObjectives(self, objectivesList):
        for objective in objectivesList:
            if self.collideCheck(objective):
                return objective
        return None

    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    def draw(self, screen, counter):  # render
        p = self.position.asInt()
        # p=adjust_position = self.position - CustomVector(cell_side_midlle, cell_side_midlle)

        # pygame.draw.circle(screen, 'yellow', p, self.radius)

        if self.direction == -2:  # right looking
            screen.blit(player_images[counter // 5], (p[0] - cell / 4, p[1] - cell / 4))

        if self.direction == 2:  # left
            screen.blit(
                pygame.transform.flip(player_images[counter // 5], True, False),
                (p[0] - cell / 4, p[1] - cell / 4),
            )

        if self.direction == 1:  # up
            screen.blit(
                pygame.transform.rotate(player_images[counter // 5], 90),
                (p[0] - cell / 4, p[1] - cell / 4),
            )

        if self.direction == -1:  # down
            screen.blit(
                pygame.transform.rotate(player_images[counter // 5], 270),
                (p[0] - cell / 4, p[1] - cell / 4),
            )
        if self.direction == 0:  # stop
            screen.blit(pac0, (p[0] - cell / 4, p[1] - cell / 4))


############################################## Ghosts
class Ghost(MovingObject):
    def __init__(self, node, pacman=None, blinky=None):
        MovingObject.__init__(self, node)
        self.name = ghosts_indicator
        self.point = 200
        self.goal = CustomVector()
        self.moving_method = self.toTargetDirection
        self.pacman = pacman
        self.mode = SwitchMode(self)
        self.blinky = blinky
        self.homeNode = node
        self.image = None
        self.get_angry = False
        # self.default_image = None
        # self.scared_image = scared_icon
        # self.dead_image = dead_icon
        # self.image = self.default_image

    def reset(self):
        MovingObject.reset(self)
        self.point = 200
        self.moving_method = self.toTargetDirection
        # self.image = self.default_image

    def update(self, delta_t):
        self.mode.update(delta_t)
        if self.mode.current_mode is scattering_mode:
            # if self.mode.current is scattering_mode:
            self.scatter()
            # self.image = self.default_image
        elif self.mode.current_mode is chasing_mode:
            # elif self.mode.current is chasing_mode:
            self.chase()
            # self.image = self.default_image
        MovingObject.update(self, delta_t)

    def scatter(self):
        self.goal = CustomVector()
        # self.image = self.default_image

    def chase(self):
        self.goal = self.pacman.position
        # self.image = self.default_image

    def respawn(self):
        self.goal = self.respawn_target.position
        # self.image = self.dead_image

    def setRespawnTarget(self, node):
        self.respawn_target = node

    def startRespawning(self):
        # self.homeNode.allowHomeAccess(self)
        self.mode.setRespawnMode()
        if self.mode.current_mode == respawning_mode:
            self.setSpeed(150)
            self.moving_method = self.toTargetDirection
            self.respawn()
        # self.image = self.dead_image

    def gets_angry(self, counter):
        self.get_angry = True
        self.setSpeed(self.basic_speed + 15)
        self.image = dead[0]  # [counter // 5]
        self.radius = 15
        # self.goal = self.pacman.position

    def startScaring(self):
        self.mode.setScaredMode()
        if self.mode.current_mode == scared_mode:
            self.setSpeed(self.basic_speed // 2)
            self.moving_method = self.randomMove
            # self.image = self.scared_image

    def startNormalMode(self):
        self.setSpeed(self.basic_speed)
        self.moving_method = self.toTargetDirection
        self.homeNode.denyAccess(down_direction, self)

        # self.homeNode.removePath(down_direction, self)
        # self.homeNode.denyHomeAccess(self)
        # self.image = self.default_image

    def draw(self, screen, img):
        if self.visible:
            p = self.position.asInt()

            if self.mode.current_mode == scared_mode:
                screen.blit(scared_icon, (p[0] - cell / 4, p[1] - cell / 4))
            elif self.mode.current_mode == respawning_mode:
                screen.blit(dead_icon, (p[0] - cell / 4, p[1] - cell / 4))
            else:
                screen.blit(img, (p[0] - cell / 4, p[1] - cell / 4))


############################################## Funky
class Funky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = funky
        self.color = "green"
        self.image = funky_img

    def scatter(self):
        self.goal = CustomVector(0, window_height)

    def chase(self):
        distance_squared = (self.pacman.position - self.position).magnitudeSquared()
        if distance_squared <= (cell_side_size * 8) ** 2:
            self.scatter()
        else:
            self.goal = self.blinky.position


############################################## Sue
class Magenda(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = sue
        self.color = "purple"
        self.image = sue_img

    def scatter(self):
        self.goal = CustomVector(window_width, 0)

    def chase(self):
        distance_squared = (self.pacman.position - self.position).magnitudeSquared()
        if distance_squared <= (cell_side_size * 8) ** 2:
            self.scatter()
        else:
            self.goal = self.pacman.position


############################################## Blinky
class Blinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = blinky_indicator
        self.color = "red"
        self.image = blinky_icon
        # self.default_image = blinky_icon

    def scatter(self):
        self.goal = CustomVector(window_width, 0)

    def chase(self):
        self.goal = self.pacman.position


############################################## Pinky
class Pinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = pinky_indicator
        self.color = "pink"
        self.image = pinky_icon
        # self.default_image = pinky_icon

    def scatter(self):
        self.goal = CustomVector(0, 0)

    def chase(self):
        self.goal = (
            self.pacman.position
            + self.pacman.directions[self.pacman.direction] * cell_side_size * 2
        )


############################################## Inky
class Inky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = inky_indicator
        self.color = "teal"
        self.image = inky_icon
        # self.default_image = inky_icon

    def scatter(self):
        self.goal = CustomVector(window_width, window_height)

    def chase(self):
        vector_1 = (
            self.pacman.position
            + self.pacman.directions[self.pacman.direction] * cell_side_size * 2
        )
        vector_2 = (vector_1 - self.blinky.position) * 2
        self.goal = self.blinky.position + vector_2


############################################## Clyde
class Clyde(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = clyde_indicator
        self.color = "orange"
        self.image = clyde_icon
        # self.default_image = clyde_icon

    def scatter(self):
        self.goal = CustomVector(0, window_height)

    def chase(self):
        distance_squared = (self.pacman.position - self.position).magnitudeSquared()
        if distance_squared <= (cell_side_size * 8) ** 2:
            self.scatter()
        else:
            self.goal = self.pacman.position


############################################## Ghost Group
class GhostGroup(object):
    def __init__(self, node, pacman, level=0):
        if level == 0:
            self.blinky = Blinky(node, pacman)
            self.pinky = Pinky(node, pacman)
            self.inky = Inky(node, pacman, self.blinky)
            self.clyde = Clyde(node, pacman)
            self.ghosts = [self.blinky, self.pinky, self.inky, self.clyde]
        if level == 1:
            self.blinky = Blinky(node, pacman)
            self.sue = Magenda(node, pacman)
            self.inky = Inky(node, pacman, self.blinky)
            self.clyde = Clyde(node, pacman)
            self.ghosts = [self.blinky, self.sue, self.inky, self.clyde]
        if level == 2:
            self.blinky = Blinky(node, pacman)
            self.sue = Magenda(node, pacman)
            self.inky = Inky(node, pacman, self.blinky)
            self.funky = Funky(node, pacman, self.sue)
            self.ghosts = [self.blinky, self.sue, self.inky, self.funky]

    def __iter__(self):
        return iter(self.ghosts)

    def update(self, delta_t):
        for ghost in self:
            ghost.update(delta_t)

    def startScaring(self):
        for ghost in self:
            ghost.startScaring()
        self.resetPoint()

    def setRespawnTarget(self, node):
        for ghost in self:
            ghost.setRespawnTarget(node)

    def upgradePoint(self):
        for ghost in self:
            ghost.point *= 2

    def resetPoint(self):
        for ghost in self:
            ghost.point = 200

    def reset(self):
        for ghost in self:
            ghost.reset()

    def render(self, screen):
        for ghost in self:
            ghost.draw(screen, ghost.image)

    def hide(self):
        for ghost in self:
            ghost.visible = False

    def show(self):
        for ghost in self:
            ghost.visible = True


############################################## Mode
class Modes(object):
    def __init__(self):
        self.timer = 0
        self.scatter()

    def update(self, delta_t):
        self.timer += delta_t
        if self.timer >= self.time:
            if self.mode is scattering_mode:
                self.chase()
            elif self.mode is chasing_mode:
                self.scatter()

    def scatter(self):
        self.mode = scattering_mode
        self.time = 7
        self.timer = 0

    def chase(self):
        self.mode = chasing_mode
        self.time = 20
        self.timer = 0


############################################## Mode Controller
class SwitchMode(object):
    def __init__(self, single_object):
        self.timer = 0
        self.time = None
        self.mainmode = Modes()
        self.current_mode = self.mainmode.mode
        self.single_object = single_object

    def update(self, delta_t):
        self.mainmode.update(delta_t)
        if self.current_mode is scared_mode:
            self.timer += delta_t
            if self.timer >= self.time:
                self.time = None
                self.single_object.startNormalMode()
                self.current_mode = self.mainmode.mode
        elif self.current_mode in [scattering_mode, chasing_mode]:
            self.current_mode = self.mainmode.mode

        if self.current_mode is respawning_mode:
            # self.nodes.allowHomeAccess(self)
            if self.single_object.node == self.single_object.respawn_target:
                self.single_object.startNormalMode()
                self.current_mode = self.mainmode.mode
                # self.nodes.denyHomeAccess(self)

    def setScaredMode(self):
        if self.current_mode in [scattering_mode, chasing_mode]:
            self.timer = 0
            self.time = 7
            self.current_mode = scared_mode
        elif self.current_mode is scared_mode:
            self.timer = 0

    def setRespawnMode(self):
        if self.current_mode is scared_mode:
            self.current_mode = respawning_mode
