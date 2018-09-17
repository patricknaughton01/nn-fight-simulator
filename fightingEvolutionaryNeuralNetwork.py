#####################################
# An evolution simulator that teaches
# an AI how to fight a randomly moving
# opponent. Either side can win
# by jumping on the other's head
#
# Date: 10.1.2018
#
# Author: Patrick Naughton
#####################################
import math
import random

import pygame
import pygame.font
import sys
from evolutionaryNeuralNetwork import Evolution
from evolutionaryNeuralNetwork import NeuralNetwork

pygame.init()
pygame.font.init()
width = 900
height = 480
f = open("generations.log", "w")


def main():
    ev = FightEvolver(100, [15, 15, 15, 3], 20, 20, 0.55)
    screen = Window(width, height, [255, 178, 102], 60)
    screen.initMainScreen()
    screen.setCallback(0, ev.doGeneration)
    screen.setCallback(1, ev.doGeneration)
    screen.setCallback(7, ev.doGeneration)
    screen.setCallback(8, ev.massExtinction)
    while True:
        screen.loop(ev)


class Window:
    def __init__(self, width, height, backgroundColor, fps):
        self.width = width
        self.height = height
        self.backgroundColor = backgroundColor
        self.surface = pygame.display.set_mode([width, height])
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.buttons = []
        self.graphs = []
        self.textBoxes = []

    def initMainScreen(self):
        spacing = 20
        buttonStepByStep = self.makeButton(230, 50, [0, 0], "Do Step By Step", None,
                                           pygame.font.Font(None, 30), (1, self))
        buttonStepByStep.rightAlign(self.surface)
        self.buttons.append(buttonStepByStep)
        buttonASAP = self.makeButton(230, 50, [0, buttonStepByStep.height + buttonStepByStep.position[1] + spacing],
                                     "ASAP", None, pygame.font.Font(None, 30), (2, self))
        buttonASAP.rightAlign(self.surface)
        self.buttons.append(buttonASAP)
        generationLabel = self.makeButton(400, 60, [0, 0], "Generation: ", None, pygame.font.Font(None, 40), ())
        self.buttons.append(generationLabel)
        winLabel = self.makeButton(100, 40, [0, generationLabel.height + generationLabel.position[1] + spacing],
                                   "Wins: ", None
                                   , pygame.font.Font(None, 20), (), color=[255, 81, 81])
        lossLabel = self.makeButton(100, 40, [0, winLabel.height + winLabel.position[1] + spacing], "Losses: ",
                                    None, pygame.font.Font(None, 20), (), color=[255, 81, 81])
        fallLabel = self.makeButton(100, 40, [0, lossLabel.height + lossLabel.position[1] + spacing], "Falls: ",
                                    None, pygame.font.Font(None, 20), (), color=[255, 81, 81])
        self.buttons.extend([winLabel, lossLabel, fallLabel])
        sHeight = 50
        statusButton = self.makeButton(self.width, sHeight, [0, self.height - sHeight], "Status: ", None,
                                       pygame.font.Font(None, 40), (), color=[51, 51, 255])
        self.buttons.append(statusButton)
        buttonALAP = self.makeButton(230, 50, [0, buttonASAP.height + buttonASAP.position[1] + spacing], "ALAP",
                                     None, pygame.font.Font(None, 30), (3, self))
        buttonALAP.rightAlign(self.surface)
        self.buttons.append(buttonALAP)
        massExtinctionButton = self.makeButton(230, 50,
                                               [0, buttonALAP.height + buttonALAP.position[1] + spacing], "Extinction",
                                               None, pygame.font.Font(None, 30), [0.1, self])
        massExtinctionButton.rightAlign(self.surface)
        self.buttons.append(massExtinctionButton)
        graphSize = 150
        winGraph = Graph(winLabel.width + winLabel.position[0] + spacing,
                         winLabel.position[1], 1.75 * graphSize, graphSize, "Wins", points=[[0, 0], [0, 0]])
        self.graphs.append(winGraph)
        lossGraph = Graph(0, fallLabel.position[1] + spacing + fallLabel.height,
                          1.75 * graphSize, graphSize, "Losses", points=[[0, 0], [0, 0]])
        self.graphs.append(lossGraph)
        fallGraph = Graph(lossGraph.width + lossGraph.x + spacing,
                          lossGraph.y, 1.75 * graphSize, graphSize, "Falls", points=[[0, 0], [0, 0]])
        self.graphs.append(fallGraph)
        boxSize = 100
        winBox = TextBox(generationLabel.rect.right + spacing, generationLabel.rect.centery, boxSize, boxSize / 3, "Win Const")
        self.textBoxes.append(winBox)
        loseBox = TextBox(winBox.x, winBox.y + winBox.height + spacing, boxSize, boxSize / 3, "Lose Const")
        self.textBoxes.append(loseBox)
        fallBox = TextBox(winBox.x, loseBox.y + loseBox.height + spacing, boxSize, boxSize / 3, "Fall Const")
        self.textBoxes.append(fallBox)
        survBox = TextBox(winBox.x, fallBox.y + fallBox.height + spacing, boxSize, boxSize / 3, "Surv Const")
        self.textBoxes.append(survBox)
        timeBox = TextBox(fallGraph.x + fallGraph.width + spacing, fallGraph.y, boxSize, boxSize / 3, "Time Mul")
        self.textBoxes.append(timeBox)
        closeBox = TextBox(timeBox.x, timeBox.y + timeBox.height + spacing, boxSize, boxSize / 3, "Close Mul")
        self.textBoxes.append(closeBox)
        topBox = TextBox(timeBox.x, closeBox.y + closeBox.height + spacing, boxSize, boxSize / 3, "Top Mul")
        self.textBoxes.append(topBox)
        jumpBox = TextBox(timeBox.rect.right + spacing, closeBox.rect.top, boxSize, boxSize / 3, "Jump?")
        self.textBoxes.append(jumpBox)
        timeBox = TextBox(jumpBox.rect.left, jumpBox.rect.bottom + spacing, boxSize, boxSize / 3, "Max Time")
        self.textBoxes.append(timeBox)
        heightBox = TextBox(jumpBox.rect.right + spacing/2, jumpBox.rect.top, boxSize, boxSize / 3, "Height")
        self.textBoxes.append(heightBox)
        cutoffBox = TextBox(winBox.rect.right + spacing, winBox.rect.top, boxSize, boxSize / 3, "Cutoff")
        self.textBoxes.append(cutoffBox)


    def setCallback(self, buttonID, callback):
        self.buttons[buttonID].callback = callback

    def loop(self, evolver):
        self.surface.fill(self.backgroundColor)
        self.buttons[2].setText("Generation: " + str(evolver.generation))
        self.buttons[3].setText("Wins: " + str(evolver.wins))
        self.buttons[4].setText("Losses: " + str(evolver.losses))
        self.buttons[5].setText("Falls: " + str(evolver.falls))
        for b in self.buttons:
            b.draw(self.surface)
        for g in self.graphs:
            g.draw(self.surface)
        for t in self.textBoxes:
            t.draw(self.surface)
        try:
            evolver.winConst = float(self.textBoxes[0].text)
        except:
            evolver.winCosnt = 0
        try:
            evolver.loseConst = float(self.textBoxes[1].text)
        except:
            evolver.loseConst= 0
        try:
            evolver.fallConst= float(self.textBoxes[2].text)
        except:
            evolver.fallConst  = 0
        try:
            evolver.survConst = float(self.textBoxes[3].text)
        except:
            evolver.survConst= 0
        try:
            evolver.timeMul = float(self.textBoxes[4].text)
        except:
            evolver.timeMul = 0
        try:
            evolver.closeMul = float(self.textBoxes[5].text)
        except:
            evolver.closeMul = 0
        try:
            evolver.topMul = float(self.textBoxes[6].text)
        except:
            evolver.topMul = 0
        try:
            evolver.jump = float(self.textBoxes[7].text)
        except:
            evolver.jump = 0
        try:
            evolver.maxTime = float(self.textBoxes[8].text)
        except:
            evolver.maxTime = 1
        try:
            evolver.height2 = float(self.textBoxes[9].text)
        except:
            evolver.height2 = 0
        try:
            evolver.fitnessCutoff = float(self.textBoxes[10].text)
        except:
            evolver.fitnessCutoff = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    for button in self.buttons:
                        button.do()
                    for t in self.textBoxes:
                        t.update()
            if event.type == pygame.KEYDOWN:
                for t in self.textBoxes:
                    t.addText(event)
        self.buttons[6].setText("Status: " + evolver.status)
        pygame.display.flip()

    def makeButton(self, width, height, position, text, callback, font, args, color=[0, 255, 0]):
        return Button(width, height, position, text, callback, font, args, color=color)


class Button:
    def __init__(self, width, height, position, text, callback, font, args, textColor=[255, 255, 255],
                 color=[50, 255, 50]):
        self.width = width
        self.height = height
        self.position = position
        self.text = text
        self.textColor = textColor
        self.color = color
        if callback is None:
            self.callback = self.empty
        else:
            self.callback = callback
        self.rect = pygame.Rect(self.position[0], self.position[1], self.width, self.height)
        self.font = font
        self.textSurface = self.font.render(self.text, 0, self.textColor)
        self.textRect = pygame.Rect((0, 0), self.font.size(self.text))
        self.textRect.centerx = self.rect.centerx
        self.textRect.centery = self.rect.centery
        self.args = args

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        surface.blit(self.textSurface, self.textRect)

    def setText(self, text):
        self.text = text
        self.textSurface = self.font.render(self.text, 0, self.textColor)
        self.textRect = pygame.Rect((0, 0), self.font.size(self.text))
        self.textRect.centerx = self.rect.centerx
        self.textRect.centery = self.rect.centery

    def do(self):
        if self.rect.collidepoint(pygame.mouse.get_pos()):
            return self.callback(*self.args)

    def rightAlign(self, surface):
        self.position[0] = surface.get_width() - self.width
        self.updatePos()

    def leftAlign(self):
        self.position[0] = 0
        self.updatePos()

    def topAlign(self):
        self.position[1] = 0
        self.updatePos()

    def bottomAlign(self, surface):
        self.position[1] = surface.get_height() - self.height
        self.updatePos()

    def updatePos(self):
        self.rect.top = self.position[1]
        self.rect.left = self.position[0]
        self.textRect.centerx = self.rect.centerx
        self.textRect.centery = self.rect.centery

    def empty(self):
        pass


class TextBox:
    def __init__(self, x, y, width, height, label, color=(255, 255, 255),
                 textColor=(0, 0, 0), maxChars=10, focusColor=(0, 0, 255), textSize=20):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.color = color
        self.textColor = textColor
        self.maxChars = maxChars
        self.hasFocus = False
        self.focusColor = focusColor
        self.textSize = textSize
        self.text = ""
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, screen):
        font = pygame.font.Font(None, self.textSize)
        surf = font.render(self.text, 0, self.textColor)
        pygame.draw.rect(screen, self.color, self.rect)
        rect = pygame.Rect((0, 0), font.size(self.text))
        rect.centerx = self.x + self.width / 2
        rect.centery = self.y + self.height / 2
        screen.blit(surf, rect)
        surf = font.render(self.label, 0, self.textColor)
        rect = pygame.Rect((0, 0), font.size(self.label))
        rect.centerx = self.x + self.width / 2
        rect.bottom = self.rect.top
        screen.blit(surf, rect)
        if self.hasFocus:
            pygame.draw.rect(screen, self.focusColor, self.rect, 1)

    def addText(self, event):
        if self.hasFocus:
            if len(self.text) < self.maxChars:
                if event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:len(self.text) - 1]
                else:
                    self.text += event.unicode

    def update(self):
        if self.rect.collidepoint(pygame.mouse.get_pos()):
            self.hasFocus = True
        else:
            self.hasFocus = False


class Graph:
    def __init__(self, x, y, width, height, title, points=[], color=[255, 255, 255], lineColor=[0, 0, 0]):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.points = points
        self.color = color
        self.title = title
        self.lineColor = lineColor

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, [self.x, self.y, self.width, self.height])
        titleFont = pygame.font.Font(None, 20)
        titleSurface = titleFont.render(self.title, 0, self.lineColor)
        titleWidth = titleFont.size(self.title)[0]
        surface.blit(titleSurface, [self.x + self.width / 2 - titleWidth / 2, self.y])
        d = self.getDomain()
        dSize = d[1] - d[0]
        if dSize == 0:
            dSize = 1
        r = self.getRange()
        rSize = r[1] - r[0]
        if rSize == 0:
            rSize = 1
        displayPoints = []
        for p in self.points:
            displayPoints.append(
                [self.x + self.width * p[0] / dSize, self.y + self.height - self.height * p[1] / rSize])
        pygame.draw.lines(surface, self.lineColor, False, displayPoints)

    def getDomain(self):
        if len(self.points) > 0:
            min = self.points[0][0]
            max = self.points[0][0]
            for p in self.points:
                if p[0] < min:
                    min = p[0]
                if p[0] > max:
                    max = p[0]
            return ([min, max])

    def getRange(self):
        if len(self.points) > 0:
            min = self.points[0][1]
            max = self.points[0][1]
            for p in self.points:
                if p[1] < min:
                    min = p[1]
                if p[1] > max:
                    max = p[1]
            return ([min, max])

    def addPoint(self, point):
        if len(point) == 2:
            self.points.append(point)


class FightEvolver(Evolution):
    def __init__(self, numCreatures, creatureArchitecture, weightRange, volatilityRange, percentSurvival):
        Evolution.__init__(self, numCreatures, creatureArchitecture, weightRange, volatilityRange, percentSurvival)
        self.wins = 0
        self.losses = 0
        self.falls = 0
        self.status = "Idle"
        self.playback = 1
        # defining the fitness function
        self.winConst = 0
        self.loseConst = 0
        self.fallConst = 0
        self.survConst = 0
        self.timeMul = 1
        self.closeMul = 1
        self.topMul = 0
        # zero is player2 never jumps, one is player2 always jumps
        self.jump = 0
        self.maxTime = 15
        #jump speed of player2
        self.height2 = 0
        self.fitnessCutoff = 0

    def doGeneration(self, mode, window):
        global f
        go = True
        while go:
            self.generation += 1
            self.assessAllFitness(mode, window)
            window.graphs[0].addPoint([self.generation, self.wins])
            window.graphs[1].addPoint([self.generation, self.losses])
            window.graphs[2].addPoint([self.generation, self.falls])
            self.sortFitness()
            f.write("Best creature of generation " + str(self.generation) + " had structure: \n")
            f.write(str(self.creatureArchitecture) + " and weights:\n")
            for L in self.creatures[0].layers:
                f.write(str(L.weights) + "\n\n")
            f.write("\n****************************************************************\n")
            survivors = self.creatures[:int(len(self.creatures) * self.percentSurvival)]
            for s in range(len(survivors)-1, -1, -1):
                if self.fitnessScores[s]<self.fitnessCutoff:
                    survivors.pop(s)
            if len(survivors) == 0:
                for i in range(self.numCreatures):
                    self.creatures = []
                    self.creatures.append(NeuralNetwork(self.creatureArchitecture, self.weightRange,
                                                        self.volatilityRange - 2*random.random()*self.volatilityRange))
            else:
                ind = 0
                while len(survivors) < self.numCreatures:
                    if ind < len(survivors):
                        survivors.append(survivors[ind].reproduce())
                    else:
                        ind = 0
                self.creatures = survivors
            if not mode == 3:
                go = False
                break
            else:
                self.status = "ALAPing"
                window.loop(self)
                window.loop(self)
                if self.containsOne(pygame.key.get_pressed()):
                    self.status = "Idle"
                    go = False
                    break
        return

    def massExtinction(self, percentageSurvival, window):
        survivors = []
        if percentageSurvival < 1 and percentageSurvival > 0:
            for i in range(int(self.numCreatures * percentageSurvival)):
                ind = random.choice(range(len(self.creatures)))
                survivors.append(self.creatures.pop(ind))
        ind = 0
        while len(survivors) < self.numCreatures:
            if ind < len(survivors):
                newBorn = survivors[ind].reproduce()
                survivors.append(newBorn)
                ind += 1
            else:
                ind = 0
        self.creatures = survivors
        self.doGeneration(2, window)

    def assessAllFitness(self, mode, window):
        self.wins, self.losses, self.falls = 0, 0, 0
        self.fitnessScores = []
        if mode == 2:
            self.status = "ASAPing"
            window.loop(self)
            window.loop(self)
        for i in range(len(self.creatures)):
            self.fitnessScores.append(self.assessFitness(i, mode, window))
        self.status = "Idle"
        return

    def assessFitness(self, creatureID, mode, w):
        clock = pygame.time.Clock()
        fps = 30
        width = w.width-200
        height = w.height
        size = 10
        spacing = size * 10
        maxTime = self.maxTime
        p1Start = random.randint(0, spacing)
        p2Start = random.randint(width / 2 - spacing, width / 2 + spacing)
        player1 = Player(p1Start, 0, size, size, [255, 255, 0])
        player2 = Player(p2Start + spacing, player1.rect.bottom + spacing, size, size, [255, 0, 0])
        player2.xVel = player2.speed
        ground = Ground(0, height / 2, width, height)
        font = pygame.font.Font(None, 30)
        secondsClose = 0
        closeThresh = 20
        if not mode == 1:
            self.playback = 500
        for i in range(int(maxTime * fps)):
            clock.tick(fps * self.playback)
            input = [[player2.rect.left - player1.rect.left],
                     [player2.rect.top - player1.rect.top],
                     [player2.xVel],
                     [player2.yVel],
                     [player1.xVel],
                     [player1.yVel],
                     [ground.rect.left - player1.rect.left],
                     [ground.rect.right - player1.rect.right],
                     [player1.rect.width],
                     [player1.rect.height],
                     [player2.rect.width],
                     [player2.rect.height],
                     [player1.canJump],
                     [player2.canJump],
                     [i / fps]]
            output = self.creatures[creatureID].generateOutput(input)
            if output[0][0] > 0.5:
                player1.xVel = player1.speed
            if output[1][0] > 0.5:
                player1.xVel = -player1.speed
            if output[2][0] > 0.5:
                player1.jump()
            p1_r = player1.update([ground], height, [player2])
            if not (p1_r is None):
                if p1_r == "win":
                    self.wins += 1
                    return (self.winConst + self.timeMul / (i / fps) +
                            self.closeMul * secondsClose / fps + self.topMul * (player2.rect.top - player1.rect.top))
                elif p1_r == "loss":
                    self.losses += 1
                    return (self.loseConst + self.timeMul / (i / fps) +
                            self.closeMul * secondsClose / fps + self.topMul * (player2.rect.top - player1.rect.top))
                elif p1_r == "fall":
                    self.falls += 1
                    return (self.fallConst + self.timeMul / (i / fps) +
                            self.closeMul * secondsClose / fps + self.topMul * (player2.rect.top - player1.rect.top))
            if i % fps / 3 == 0:
                player2.xVel = -player2.xVel
            if random.random()<self.jump:
                player2.jumpSpeed = self.height2
                player2.jump()
            player2.update([ground], height, [])
            if math.fabs(player1.rect.centerx - player2.rect.centerx) < closeThresh:
                secondsClose += 1
            if mode == 1:
                w.surface.fill([0, 0, 0])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        # if key is numeric
                        if 57 >= event.key >= 48:
                            # update playback
                            self.playback = event.key - 47
                player1.draw(w.surface)
                player2.draw(w.surface)
                ground.draw(w.surface)
                w.surface.blit(font.render(str(creatureID + 1), 0, [255, 255, 255]), [100, w.height - 100])
                pygame.display.flip()
        r = player1.rect.left - player2.rect.left
        if r == 0:
            r = 1
        return (self.survConst + self.timeMul / maxTime +
                self.closeMul * secondsClose / fps + self.topMul * (player2.rect.top - player1.rect.top))

    def containsOne(self, arr):
        for i in arr:
            if i == 1:
                return (True)
        return (False)


class Ground:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = [0, 0, 255]
        self.type = "Ground"

    def draw(self, s):
        pygame.draw.rect(s, self.color, self.rect)


class Player:
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.yVel = 0
        self.xVel = 0
        self.color = color
        self.type = "Player"
        self.speed = 3
        self.jumpSpeed = 10
        self.gravity = 1
        self.canJump = False

    def update(self, collideables, bottom, harmables):
        self.updateYVel()
        self.updateX()
        for coll in collideables:
            if self.rect.colliderect(coll.rect):
                if self.xVel > 0:
                    self.rect.right = coll.rect.left
                elif self.xVel < 0:
                    self.rect.left = coll.rect.right
                self.xVel = 0
        self.updateY()
        for coll in collideables:
            if self.rect.colliderect(coll.rect):
                if self.yVel > 0:
                    self.canJump = True
                    self.rect.bottom = coll.rect.top
                elif self.yVel < 0:
                    self.rect.top = coll.rect.bottom
                self.yVel = 0
        for h in harmables:
            if self.rect.colliderect(h.rect):
                if self.yVel - h.yVel > 0 and self.rect.top < h.rect.top:
                    return ("win")
                else:
                    return ("loss")
        if self.rect.top > bottom:
            return ("fall")

    def jump(self):
        if self.canJump:
            self.yVel -= self.jumpSpeed
            self.canJump = False

    def draw(self, s):
        pygame.draw.rect(s, self.color, self.rect)

    def updateX(self):
        self.rect.left += self.xVel

    def updateY(self):
        self.rect.top += self.yVel

    def updateYVel(self):
        self.applyGravity()

    def applyGravity(self):
        self.yVel += self.gravity


if __name__ == "__main__":
    main()

f.close()
