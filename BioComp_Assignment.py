import random
from statistics import mean
import matplotlib.pyplot as plt
from operator import attrgetter

dataloc = "datasets/data2.txt" #the location of the dataset
condLength = 6 #the number of bits in the condition
outLength = 1 #the number of bits in the output

N = 420 #number of bits in the string
P = 700 #population size (no of individuals in the population)
nGen = 500 #number of generations
mutRate = 0.0008 #mutation rate 1/N
crossoverRate = 0.95 #rate of crossover
nSlice = N / (condLength + outLength) #how many times should the gene should be split to compare to the rule. should be N / (condLength + outLength)
maxFitness = 60 #stop searching when this fitness value is reached
wildcards = True #whether or not to use wildcards
elitism = True #whether to replace worst individual with best one each generation
selection = "tournament" #selection type "roulette" or "tournament"

class rule():
    condition = []
    output = []

    def __init__(self, condition=[0]*condLength, output=[0]*outLength): 
        self.condition = condition
        self.output = output
        
    def getInputLength(self):
        return len(self.condition)
    
    def getOutputLength(self):
        return len(self.output)

    def setCondition(self, cond):
        self.condition = cond

    def setOutput(self, out):
        self.output = out

    def printRule(self):
        print("Rule with condition: ", self.condition, "and output", self.output)



class individual():
    gene = [] 
    fitness = 0

    def __init__(self):
        self.gene = []
        self.fitness = 0

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and self.gene == other.gene)
        
    def randomiseGene(self):
        for i in range (1, N+1):
            if not wildcards:
                c = c = random.randint(0, 1)
                self.gene.append(c)
            else: #have a possibility of 2 being appended to the gene
                if i % (condLength+1) == 0:
                    c = random.randint(0, 1)
                    self.gene.append(c)
                else:
                    c = random.randint(0, 2)
                    self.gene.append(c)

    def setGene(self, gene):
        self.gene = gene

    def printGene(self):
         print(self.gene)
        
    def printFitness(self):
        print(self.fitness)

    def fitnessFunction(self, rulebase):
        sliceList = self.sliceGene()
        count = 0
        for r in rulebase: 
            index = 0
            for slice in sliceList:
                k = 0
                while k < condLength:
                    if r.condition[k] == slice[k] or slice[k] == 2:
                        k += 1
                    else:
                        index +=1
                        break
                else:
                    if slice[condLength] == r.output[outLength-1]:
                        count += 1
                    index += 1
                    break
        self.fitness = count

    def sliceGene(self):
        avg = len(self.gene) / float(nSlice)
        list = []
        last = 0.0
        while last < len(self.gene):
            list.append(self.gene[int(last):int(last + avg)])
            last += avg
        return list
        
def loadDataset(dataloc):
    rds = []
    f = open(dataloc, "r")
    for line in f:
        line = str(line.rstrip()).split(" ")
        #print(line)
        rds.append(line)
    return rds #raw data

def buildRulebase(dataset):
    rulebase = []
    for data in dataset:
        r = rule()
        cond = []
        for b in data[0]:
            cond.append(int(b))
        r.setCondition(cond)
        r.setOutput([int(data[1])])
        rulebase.append(r)
    return rulebase

def createInitalPopulation(): #create P number of individuals and append them to the list.
    pop = []
    for x in range(0, P):
        i = individual()
        pop.append(i)
    return pop

def randomisePopulation(pop): #function to set up the intial random genes in the population
    for i in pop:
        i.randomiseGene()
        #i.printGene()

def recalculateAllFitness(pop, rulebase): #run the fitness function for all members in the pop
    for i in pop:
        i.fitnessFunction(rulebase)

def randomIndividual(pop): #function to pick and return a random individual from a given list
    indiv = random.choice(pop)
    return indiv

def selectOneRW(pop):
        max = sum([c.fitness for c in pop])
        pick = random.uniform(0, max)
        current = 0
        for i in pop:
            current += i.fitness
            if current > pick:
                return i

def selectWinners(pop): #function to select P number of winners by comparing two and selecting the one with the highest fitness or a random one if equal
    if selection == "tournament":
        winners = []
        while len(winners) != P:
            i1 = randomIndividual(pop)
            i2 = randomIndividual(pop)
            if i1.fitness > i2.fitness and i1 != i2:
                winners.append(i1)
            elif i1.fitness == i2.fitness and i1 != i2:
                winners.append(random.choice([i1,i2]))
            elif i2.fitness > i1.fitness:
                winners.append(i2)
        return winners
    elif selection == "roulette":
        winners = []
        while len(winners) != P:
            i1 = selectOneRW(pop)
            i2 = selectOneRW(pop)
            if i1.fitness > i2.fitness and i1 != i2:
                winners.append(i1)
            elif i1.fitness == i2.fitness and i1 != i2:
                winners.append(random.choice([i1,i2]))
            elif i2.fitness > i1.fitness:
                winners.append(i2)
    return winners

def mutateIndividual(indiv): #perform mutation in all bits in an individuals gene
    if wildcards:
        for index, b in enumerate(indiv.gene):
            if random.random() < mutRate:
                if b == 1:
                    if index % (condLength+1) == 0:
                        indiv.gene[index] = random.choice([0,2])
                elif b == 0:
                    if index % (condLength+1) == 0:
                        indiv.gene[index] = random.choice([1,2])
                elif b == 2:
                    indiv.gene[index] = random.choice([0,1])
    elif not wildcards:
        for index, b in enumerate(indiv.gene):
            if random.random() < mutRate:
                if b == 1:
                    if index % (condLength+1) == 0:
                        indiv.gene[index] = 0
                elif b == 0:
                    if index % (condLength+1) == 0:
                        indiv.gene[index] = 1


def doCrossover(pop):
    motherlist = pop[::2] #get every member at even positions
    fatherlist = pop[1::2] #get every member at odd positions
    postCrossoverChilden = [] # a list to hold the children after crossover
    for i in range(0, int(P/2)):
        parent1 = motherlist[i]
        parent2 = fatherlist[i]
        if random.random() < crossoverRate:
            pivotpoint = random.randint(0,N)
            while pivotpoint == 0 or pivotpoint == N: #make sure pivotpoint is not 0 or N
                pivotpoint = random.randint(0,N)
            #print("pivotpoint", pivotpoint, "p1gene", motherlist[i].fitness, motherlist[i].fitness, "p2gene", fatherlist[i].fitness, fatherlist[i].fitness)
            child1Gene = fatherlist[i].gene[:pivotpoint] + motherlist[i].gene[pivotpoint:]
            child1 = individual()
            child1.setGene(child1Gene)
            child1.fitnessFunction(rulebase)
            postCrossoverChilden.append(child1)
            child2Gene = motherlist[i].gene[:pivotpoint] + fatherlist[i].gene[pivotpoint:]
            child2 = individual()
            child2.setGene(child2Gene)
            child2.fitnessFunction(rulebase)
            postCrossoverChilden.append(child2)
        else:
            postCrossoverChilden.append(parent1)
            postCrossoverChilden.append(parent2)
    return postCrossoverChilden

def findGoldenBaby(pop): #this function is specfic to the fitness function used
    for i in pop:
        if i.fitness == maxFitness:
            print("  GoldenBaby: Individual with fitness " + str(i.fitness) + ", found! Quit searching")
            return i
    return None

def findAllTimeBest(pop, previous):
    if previous == None:
        previous = pop[0]
    best = previous
    for i in pop:
        if i.fitness > best.fitness:
            best = i
    return best

def replaceWorstWithBest(pop, best):
    if best != None:
        worst = min(pop, key=attrgetter('fitness')) #get the individual with the lowest fitness value from the list
        if worst.fitness != best.fitness:
            if not best in pop:
                pop = [best if x==worst else x for x in pop] #replace the worst with the best
                print("  Elitism: Replaced worst individual with fitness", worst.fitness, "with best idividual of fitness", best.fitness)
            else:
                print("  Elitism: population already contains the best individual")
        else:
            print("  Elitism: Population contians all same fitness values!")
    return pop

def showPlot(mean, best):
    print("  Opening graph...")
    plt.plot(best)
    #plt.legend(['Best individual'], ['mean fitness'])
    plt.plot(mean)
    plt.legend(['Best fitness in population', 'Mean fitness of population'])
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()

#Import dataset and build rulebase
ds = loadDataset(dataloc)
rulebase = buildRulebase(ds)


def runGA():
    #Lists to store data for plotting the graph
    meanPlot = []
    bestPlot = []

    #Lists to store population
    winners = []
    population = []
    goldenBaby = None
    allTimeBest = None
    generation = 0

    #Create population with random genes and generate their fitness values
    population = createInitalPopulation()
    randomisePopulation(population)
    recalculateAllFitness(population, rulebase)

    #Print some information to the user
    print("  The average fitness for the initial population is", mean(i.fitness for i in population))
    if wildcards: print("  Wildcards enabled.")
    if elitism: print("  Elitism enabled.")
    if selection == "roulette" or selection == "tournament": print("  Selection method: ", selection)

    while generation <= nGen: #stop the loop once fitness N is reached
        if goldenBaby == None:
            recalculateAllFitness(population, rulebase)
            goldenBaby = findGoldenBaby(population)

        print()
        print("========Generation",str(generation)+"========")
        print()

        #Perform selection
        winners = selectWinners(population)
        if elitism: winners = replaceWorstWithBest(population, allTimeBest)

        #Perform crossover
        population = doCrossover(winners)
        #print("  mean fitness after crossover is", mean(i.fitness for i in population))

        #Perform mutation
        allTimeBest = findAllTimeBest(population, allTimeBest)
        for i in population:
            mutateIndividual(i)
        recalculateAllFitness(population, rulebase)
        #print("  mean fitness after mutation is", mean(i.fitness for i in population))

        #Gather some data
        if goldenBaby == None:
            goldenBaby = findGoldenBaby(population)
                
        if allTimeBest != None:
            print("  The fittest individual this generation has a fitness value of", allTimeBest.fitness)
            bestPlot.append(allTimeBest.fitness)

        meanVal = mean(i.fitness for i in population)
        print("  The average fitness value for this generation is", meanVal)
        meanPlot.append(meanVal)

        if goldenBaby != None:
            print("  Optimum solution found in",generation,"generations.")
            p = str(goldenBaby.gene)
            p = p.strip("[,]")
            p = p.replace(', ', '')
            print("it's gene is", p)
            #bestPlot.append(goldenBaby.fitness)
            break;  

        #Finish generation
        generation += 1

    showPlot(meanPlot, bestPlot)



def testFunc():
    population = createInitalPopulation()
    randomisePopulation(population)
    population[0].fitnessFunction(rulebase)
    population[0].printFitness()

runGA()
#testFunc()


