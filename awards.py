import pygame
from pygame.locals import *
from vector import Vectors
#from constants import *
from random import randint
from behavior import ModeController



cellw = 16   #TILEWIDTH
cellh= 16   #TILEHEIGHT
row = 36   #NROWS
col = 28  #NCOLS
w = col*cellw #SCREENWIDTH
h =row * cellh #SCREENHEIGHT = NROWS*TILEHEIGHT
screen = (w,h)
#SCREENSIZE = (SCREENWIDTH, SCREENHEIGHT)

stop = 0
up = 1 
down =-1 
left = 2 
right =-2 

pacman=0
SCATTER = 0
CHASE = 1
FREIGHT = 2
SPAWN = 3
powerdot=4

blinky = 4
pinky = 5
inky = 6
clyde = 7

gold = 99
bomb = 98
shield = 97


gold_img=pygame.transform.scale(pygame.image.load(f'assets/gold.png'),(2*cellw,2*cellw))
#funky
bomb_img=pygame.transform.scale(pygame.image.load(f'assets/bomb.png'),(2*cellw,2*cellw))
shield_img=pygame.transform.scale(pygame.image.load(f'assets/shield.png'),(2*cellw,2*cellw))

class Award(object):
    def __init__(self, node, level=0):
        self.node = node
        self.name = None
        self.color = None
        self.radius = int(8 * cellw / 16)
        self.collideRadius = int(4 * cellh / 16)
        self.lifespan = 20
        self.timer = 0
        self.destroy = False
        self.visible = True
        #self.point = 500 + level*100
        self.point = None
        self.middleOfNodes(right)
        #self.position = self.node.position
    
    def update(self, time):
        self.timer += time
        if self.timer >= self.lifespan:
            self.destroy = True

    def middleOfNodes(self, direction):
        if self.node.neighbors[direction] is not None:
            self.target = self.node.neighbors[direction]
            self.position = (self.node.position + self.target.position) / 2.0

    def reset(self):
        self.visible = True



class Gold(Award):
    def __init__(self, node, level=0):
        Award.__init__(self, node, level=0)
        self.name = gold
        self.color = 'yellow'
        #self.point = 500 + level*100
        self.point = 200

    def render(self, screen):
            if self.visible:
                p = self.position.asInt()
                screen.blit(gold_img,(p[0]-cellw/2,p[1]-cellh/2))
                #pygame.draw.circle(screen, self.color, p, self.radius)



class Bomb(Award):
    def __init__(self, node, level=0):
        Award.__init__(self, node, level=0)
        self.name = bomb
        self.color = 'red'
        #self.point = 500 + level*100
        self.point = 5

    def render(self, screen):
            if self.visible:
                p = self.position.asInt()
                screen.blit(bomb_img,(p[0]-cellw/2,p[1]-cellh/2))
                #pygame.draw.circle(screen, self.color, p, self.radius)

class Shield(Award):
    def __init__(self, node, level=0):
        Award.__init__(self, node, level=0)
        self.name = shield
        self.color = 'green'
        #self.point = 500 + level*100
        self.point = 0

    def render(self, screen):
            if self.visible:
                p = self.position.asInt()
                screen.blit(shield_img,(p[0]-cellw/2,p[1]-cellh/2))
                #pygame.draw.circle(screen, self.color, p, self.radius)