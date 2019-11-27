import random
from statistics import mean
import matplotlib.pyplot as plt
from operator import attrgetter


N = 70 #number of bits in the string
P = 150 #population size (no of individuals in the population)
nGen = 100 #number of generations
mutRate = 0.2 #mutation rate 1/N
nSlice = 10 #how many times should the gene should be split to compare to the rule. should be N / (Size of condLength + outLength)
maxFitness = 60 #stop searching when this fitness value is reached

dataloc = "datasets/data1.txt" #the location of the dataset
wildcards = True #whether or not to use wildcards
condLength = 6 #the number of bits in the condition
outLength = 1 #the number of bits in the output

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

    def compare(self, slice): #returns true if cond and output matches returns false if not
        if not wildcards:
            cond = str()
            out = None
            for i in range(0,len(slice)-outLength):
                cond += str(slice[i])
            out = slice[-1] #output
            if cond == self.condition:
                if int(out) == int(self.output):
                    return True
            return False
        else:
            #match condition
            match = True
            for i in range(0,condLength):
                if (int(slice[i]) == int(self.condition[i]) or int(slice[i]) == 2) and match == True:
                    #print("bitmatch")   
                    pass             
                else:
                    match = False
                    #print("bitFAIL")
            #match output
            oi = 0
            if not condLength == condLength+(outLength-1):
                for i in range(condLength, condLength+(outLength-1)):
                    print(i)
                    if (int(slice[i]) == int(self.output[oi]) and match == True):
                        pass
                    else: 
                        match = False
                    oi+=1
                return match
            else:
                if (int(slice[condLength]) == int(self.output[oi]) and match == True):
                        pass
                else: 
                    match = False
                    oi+=1
                return match



                    
                




class individual():
    gene = [] 
    fitness = 0

    def __init__(self):
        self.gene = []
        self.fitness = 0
        
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
            found = False
            for slice in sliceList:
                if r.compare(slice):
                    found = True
                elif not r.compare(slice):
                    found = False
                if found == True:
                    count += 1
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
        r.setCondition(data[0])
        r.setOutput(data[1])
        #r.printRule() #prints the rule
        #print(r.compare(['0','0','0','0','0','0','0'])) #test
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

def selectWinners(pop): #function to select P number of winners by comparing two and selecting the one with the highest fitness or a random one if equal
    winners = []
    while len(winners) != P:
        i1 = randomIndividual(pop)
        i2 = randomIndividual(pop)
        if i1.fitness > i2.fitness:
            winners.append(i1)
        elif i1.fitness == i2.fitness:
            winners.append(random.choice([i1,i2]))
        elif i2.fitness > i1.fitness:
            winners.append(i2)
    return winners

def mutateIndividual(indiv): #perform mutation in all bits in an individuals gene
    if not wildcards:
        for index, b in enumerate(indiv.gene):
            if random.random() < mutRate:
                if b == 1:
                    if index % (condLength+1) == 0:
                        indiv.gene[index] = random.choice([0,2])
                if b == 0:
                    if index % (condLength+1) == 0:
                        indiv.gene[index] = random.choice([1,2])

def doCrossover(pop):
    motherlist = pop[::2] #get every member at even positions
    fatherlist = pop[1::2] #get every member at odd positions
    postCrossoverChilden = [] # a list to hold the children after crossover
    for i in range(0, int(P/2)):
        parent1 = motherlist[i]
        parent2 = fatherlist[i]
        pivotpoint = random.randint(0,N)
        while pivotpoint == 0 or pivotpoint == N: #make sure pivotpoint is not 0 or N
            pivotpoint = random.randint(0,N)
        #print("pivotpoint", pivotpoint, "p1gene", motherlist[i].gene, "p2gene", fatherlist[i].gene)
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
        i.fitnessFunction(rulebase)
        if i.fitness > previous.fitness:
            best = i
    return best

def replaceWorstWithBest(pop, best):
    if best != None:
        worst = min(pop, key=attrgetter('fitness')) #get the individual with the lowest fitness value from the list
        pop = [best if x==worst else x for x in pop] #replace the worst with the best
        print("  replaced worst individual with fitness", worst.fitness, "with best idividual of fitness", best.fitness)

def showPlot(mean, best):
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

    #Create population with random genes
    population = createInitalPopulation()
    randomisePopulation(population)
    generation += 1

    #Generate their fitness values
    recalculateAllFitness(population, rulebase)
    print("  The average fitness for the initial population is", mean(i.fitness for i in population))

    while generation <= nGen: #stop the loop once fitness N is reached
        if goldenBaby == None:
            recalculateAllFitness(population, rulebase)
            goldenBaby = findGoldenBaby(population)

        print()
        print("========Generation",str(generation)+"========")
        print()
        #Perform selection
        winners = selectWinners(population)
        #print("  mean fitness after selection is", mean(i.fitness for i in winners))

        #Perform crossover
        replaceWorstWithBest(population, allTimeBest)
        population = doCrossover(winners)
        #print("  mean fitness after crossover is", mean(i.fitness for i in population))

        #Perform mutation
        for i in population:
            mutateIndividual(i)
        recalculateAllFitness(population, rulebase)
        #print("  mean fitness after mutation is", mean(i.fitness for i in population))

        

        #Gather some data
        if goldenBaby == None:
            recalculateAllFitness(population, rulebase)
            goldenBaby = findGoldenBaby(population)

             
        
        recalculateAllFitness(population, rulebase)
        allTimeBest = findAllTimeBest(population, allTimeBest)
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

runGA()

def testFunc():
    population = createInitalPopulation()
    randomisePopulation(population)

#testFunc()


