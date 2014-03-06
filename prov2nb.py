import rdflib
import logging
import json
import collections

logging.basicConfig()
prompt_number = 1

###########################################################################
# SPARQL QUERIES
###########################################################################

def queryThePlan(g):
    
    qPlan = """
    	    SELECT ?agent ?aID ?plan ?pID WHERE { 
    			?pID a prov:Plan .
			?pID rdfs:label ?plan .
    			?aID a prov:Agent .
			?aID rdfs:label ?agent.
			}
	    """
    
    result = ""                            
    for row in g.query(qPlan): 
	result += "\n\nAgent : " + row["agent"] + "\n\n"+ row["aID"]
	result += "\n\nPlan  : " + row["plan"] + "\n\n"+ row["pID"]



    return result

# Getting inputs which are also datasets
def queryAllDataset(g):
    qData = """
    	    SELECT ?input ?value WHERE {
			?input  a  prov:Entity .
			?input  a  d2s:Dataset .
			?input  d2s:value ?value .
			?activity prov:used ?input .
			?activity rdfs:label ?actLabel .
			} ORDER BY ?actLabel ?activity ?input 
	    """
    result = g.query(qData)

    return result

def queryAllModules(g):
    qData = """
    	    SELECT DISTINCT ?module ?moduleInstance 
	    WHERE {
			?moduleInstance rdfs:label ?module .
			?moduleInstance	a prov:Activity .
	    } 
	    ORDER BY ?module ?moduleInstance
	    """
    result = g.query(qData)

    return result

def queryActivityInputOutput(g):
    qAIO = """
           SELECT ?activity ?output ?input ?inpValue ?outValue WHERE {
	              ?activity prov:used ?input .
		      ?output   prov:wasGeneratedBy ?activity .
		      ?activity rdfs:label ?moduleLabel .
		      ?activity a prov:Activity .
		      ?input a prov:Entity .
		      ?output a prov:Entity .
		      ?input d2s:value ?inpValue .
		      ?output d2s:value ?outValue .
  
		      } ORDER BY ?moduleLabel ?activity ?output
	   """
    result = g.query(qAIO)
    return result


def queryAllOutputs(g):
    qData = """
    	    SELECT ?output ?outputLabel ?value ?module ?moduleInstance WHERE {
	    		?output prov:wasGeneratedBy ?moduleInstance . 
			?output rdfs:label ?outputLabel .
			?moduleInstance	a prov:Activity .
			?moduleInstance rdfs:label ?module .
			?output a  prov:Entity .
			?output d2s:value ?value
			} ORDER BY ?module ?moduleInstance ?outputLabel
	    """
    result = g.query(qData)

    return result


def queryAllActivities(g):
    qActivity = """
 	SELECT ?activity ?startTime ?endTime WHERE {
	    ?activity a prov:Activity .
	    ?activity rdfs:label ?l .
	    ?activity prov:startedAtTime ?startTime .
	    ?activity prov:endedAtTime ?endTime
	} ORDER BY ?l
    """

    result = g.query(qActivity)

    return result

###########################################################################
# Ipython Notebook cells generation code 
###########################################################################


def createHeaderCell(text, level, metadata={}):
    result = {}
    result["cell_type"] = "heading"
    result["level"] = level
    result["source"] = [text]
    result["metadata"] = metadata 
    return result

def createMarkdownCell(md_text, metadata={}):
    result = {}
    result["cell_type"] = "markdown"
    result["source"] = [md_text]
    result["metadata"] = metadata
    return result

def createCodeCell(code_input, language="python", metadata={}, outputs=[], collapsed="false"):
    global prompt_number
    result = {}
    result["cell_type"] = "code"
    result["collapsed"] = collapsed 
    result["metadata"] = metadata
    result["language"] = language
    result["outputs"] = outputs
    result["input"] = code_input
    result["prompt_number"] =prompt_number
    prompt_number +=1


    return result

###########################################################################
# Cell contents generation code 
###########################################################################


def createIPYTable(records, headers, fields, shorten):
    result = """ 
from ipy_table import *
data = [ """+str(headers) +","


    for row in records:
	result +=  "["
	for i in range(len(fields)):
            if(shorten[i]):
	       result += "'" + shortenURL(row[fields[i]]) +"',"
            else:
	       result += "'" + row[fields[i]] +"',"

	result +=  "],"

    result += "]"

    result += """
make_table(data)
apply_theme('basic') """

    return result



def shortenURL(longURL):

    result = longURL.split("/")[-1]
    return result

def plotModuleOutputCode( moduleOutput):
    result = """
import matplotlib.pyplot as plt
import numpy as np

"""
    for output in moduleOutput:
	result += "\nx=np.array(range("+str(len(moduleOutput[output]))+"))"
	result += "\ny="+str(moduleOutput[output]) 
	result += "\nplt.plot(x,y,'bo')"
	result += "\nplt.title('"+output+"')"
	result += "\nplt.show()"
    return result


def groupByOutputType(outputSets):

    outputValues = {}
    for row in outputSets:
        curModule = str(row["module"])
        curOutput = str(row["outputLabel"])
        if not(curModule in outputValues):
           outputValues[curModule] = {}
        if not(curOutput in outputValues[curModule]):
           outputValues[curModule][curOutput] = []

        outputValues[curModule][curOutput].append(float(row["value"]))
    return outputValues

#Activities records are now ?activity ?input ?output ?inpValue ?outValue
def createActivityCodes(activitiesRec):
    
    #First reorganize the activities record a bit, group them by output type, and append inputs as dictionary
    activity_output = {}
    for row in activitiesRec:
 	sActivity = shortenURL(str(row["activity"]))
 	sOutput   = shortenURL(str(row["output"]))
	act_out = sActivity+"_"+sOutput

        if not(act_out in activity_output):
	   activity_output[act_out] = {}
	   activity_output[act_out]["activity"] = sActivity
	   activity_output[act_out]["input"] = {}
        
	sInput = shortenURL(str(row["input"]))
	sInpValue = str(row["inpValue"])

	sOutput = shortenURL(str(row["output"]))
	sOutValue = str(row["outValue"])

        activity_output[act_out]["input"][sInput] = sInpValue
        activity_output[act_out]["output"] = sOutput
        activity_output[act_out]["value"] = sOutValue 

    activity_output = collections.OrderedDict(sorted(activity_output.items()))
    result =[] 
    for act_out in activity_output:
        curAct = activity_output[act_out]
	curRes = "\n"+curAct["activity"]+"."+curAct["output"]+"("
        for inpKey in curAct["input"]:
		curRes += curAct["input"][inpKey]+", "
	curRes = curRes[:-2] #removing last comma
        curRes += ") = "+str(curAct["value"])
        curRes += "\n"
        result.append(curRes)
    return result

def getHideAllInputCell():
    hideMD = """
<script type="text/javascript">
    $('div.input').hide();
</script>
"""
    return hideMD

###########################################################################
# Main code 
###########################################################################

#Prepare prefix and load the main input file
provgraph = rdflib.Graph()

rdf  = rdflib.Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
rdfs = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")
d2s  = rdflib.Namespace("http://www.data2semantics.org/d2s-platform/")
prov = rdflib.Namespace("http://www.w3.org/ns/prov#")

provgraph.bind('rdf',  rdf)
provgraph.bind('rdfs', rdfs)
provgraph.bind('d2s',  d2s)
provgraph.bind('prov', prov)

# Need to give this as parameter instead of hard coding, will do this later.
provgraph.parse("prov/prov-o.ttl", format="n3")


cells = []
cells.append(createHeaderCell("Overview Report",1))
cells.append(createMarkdownCell(queryThePlan(provgraph), 3))
#cells.append(createHeaderCell("Libraries",1))


cells.append(createHeaderCell("Modules",2))
modules = queryAllModules(provgraph)
moduleCode = createIPYTable(modules, ["Module",  "Instances"], ["module","moduleInstance"], [ True, False])
cells.append(createCodeCell(moduleCode));

cells.append(createHeaderCell("Datasets",2))
dataSets = queryAllDataset(provgraph)
datasetCode = createIPYTable(dataSets, ["Datasets","Value"], ["input","value"], [False, False])
cells.append(createCodeCell(datasetCode))

cells.append(createHeaderCell("Outputs",2))

outputSets = queryAllOutputs(provgraph)
outputCode = createIPYTable(outputSets, ["Module ","Instance ", "Output", "Value"], ["module","moduleInstance", "outputLabel", "value"], [True, False, False, False])
cells.append(createCodeCell(outputCode))

cells.append(createHeaderCell("Details",1))
cells.append(createMarkdownCell("[Detailed Information](detailed-nb.ipynb)"))

#Hide all inputs
#cells.append(createMarkdownCell(getHideAllInputCell()))

cellsMap = {}
cellsMap["cells"] = cells

worksheets = []
worksheets.append(cellsMap)


metadata = {}
metadata["name"] = "Ducktape Provenance Report"


result = {}
result["metadata"] = metadata
result["nbformat"] = 3
result["nbformat_minor"] = 0
result["worksheets"] = worksheets

main = open("main-nb.ipynb","w")
main.write(json.dumps(result))
main.close()

cells = []
cells.append(createHeaderCell("Detailed Instances",2))
activities = queryAllActivities(provgraph)
activityCode = createIPYTable(activities, ["Activity", "Start", "Stop "], ["activity","startTime","endTime"], [True, False, False])
cells.append(createCodeCell(activityCode));

cells.append(createHeaderCell("Detailed Activities",1))
cells.append(createHeaderCell("Activities input output", 2))
activitiesRec = queryActivityInputOutput(provgraph)
activityIpyTableCode = createIPYTable(activitiesRec, ["Activity", "Input", "Output"], ["activity","input","output"], [True, True, True])
cells.append(createCodeCell(activityIpyTableCode, collapsed="true"));
cells.append(createHeaderCell("Activities Code ", 2))

activitiesCells = createActivityCodes(activitiesRec)

for actCell in activitiesCells:
    actCell = "#This will be pseudo code which can be used to call Ducktape for verification\n"+actCell
    cells.append(createCodeCell(actCell))

cells.append(createMarkdownCell("[Main Notebook](main-nb.ipynb)"))

cells.append(createHeaderCell("Detailed Outputs",2))
outputValues = groupByOutputType(outputSets)

for curModule in outputValues:
    cells.append(createHeaderCell(str(curModule),2))
    cells.append(createHeaderCell("Outputs",3))
    cells.append(createCodeCell(plotModuleOutputCode(outputValues[curModule])))



cellsMap = {}
cellsMap["cells"] = cells

worksheets = []
worksheets.append(cellsMap)


metadata = {}
metadata["name"] = "Detailed Ducktape Provenance Report"


result = {}
result["metadata"] = metadata
result["nbformat"] = 3
result["nbformat_minor"] = 0
result["worksheets"] = worksheets


detail = open("detailed-nb.ipynb","w")
detail.write(json.dumps(result))
detail.close()


