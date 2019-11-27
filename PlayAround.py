class rule():
    condition = []
    output = []

    def __init__(self, condition=[0,0,0,0,0,0], output=[0]): 
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
        print("Rule with input: ", self.condition, "and output", self.output)

    def compare(self, input):
        print("do stuff here")


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
        rulebase.append(r)
    return rulebase

def printRulebase(rb):
    for r in rb:
        print("rule with condition",r.condition,"gives output",r.output)

        



dataloc = "datasets/data1.txt"

dataset = loadDataset(dataloc)
rb = buildRulebase(dataset)
printRulebase(rb)