#!/usr/bin/python

import rdflib
import logging
import json
import collections

import sys, getopt

logging.basicConfig()
prompt_number = 1

###########################################################################
# SPARQL QUERIES
###########################################################################

#Give me the plan (workflow) agent (software) that execute it 
def queryThePlan(g):
    
    qPlan = """
    	    SELECT ?agent ?aID ?plan ?pID WHERE { 
    			?pID a prov:Plan .
			?pID rdfs:label ?plan .
    			?aID a prov:Agent .
			?aID rdfs:label ?agent.
			}
	    """
    
    result = g.query(qPlan)
    return result

def getPlanInfo(planRow):
    result =""
    result += "\n\nAgent : " + planRow["agent"] + "\n\n"+ planRow["aID"]
    result += "\n\nPlan  : " + planRow["plan"] + "\n\n"+ planRow["pID"]

    return result

#Getting depended artifact from the plan
def queryDependencies(g):
    qDependency = """ SELECT ?artifactID ?groupID ?versionID WHERE {
                      ?p a  prov:Plan .
                      ?p d2s:usesArtifact ?dep .
                      ?dep d2s:hasArtifactId ?artifactID .
                      ?dep d2s:hasGroupId ?groupID .
                      ?dep d2s:hasVersion ?versionID 
                  } 
                  """ 
    result = g.query(qDependency) 
    return result
   
# Getting inputs which are also datasets
def queryAllDataset(g):
    qData = """
    	    SELECT ?activityLabel ?inputLabel ?value WHERE {
                        ?input a ?inputClass .
			?inputClass rdfs:subClassOf prov:Entity .
			?input  a  d2s:Dataset .
			?input  d2s:value ?value .
                        ?inputClass rdfs:label ?inputLabel .
                        ?activity a ?activityClass .
			?activity prov:used ?input .
			?activityClass rdfs:subClassOf prov:Activity .
                        ?activityClass rdfs:label ?activityLabel .
                        MINUS { ?input prov:wasGeneratedBy ?a } .
			} ORDER BY ?activityLabel  
	    """
    result = g.query(qData)
    return result

def queryAllInputs(g):
    qData = """
    	    SELECT ?input ?value WHERE {
                        ?input a ?inputClass .
			?inputClass rdfs:subClassOf prov:Entity .
			?input  d2s:value ?value .
			?activity prov:used ?input .
			OPTIONAL { ?activity rdfs:label ?actLabel } . 
                        MINUS { ?input prov:wasGeneratedBy ?a } .
			} ORDER BY ?actLabel ?activity ?input 
	    """
    result = g.query(qData)
    return result

def queryAllModules(g):
    qData = """
    	    SELECT ?module (COUNT(?instance) as ?moduleInstance)
	    WHERE {
			?instance a ?moduleClass .
			?moduleClass rdfs:subClassOf prov:Activity .
			?moduleClass rdfs:label ?module . 
	    } 
	    GROUP BY ?moduleClass 
	    """
    result = g.query(qData)
    return result

def queryActivityInputOutput(g):
    qAIO = """
           SELECT ?activity ?output ?input ?inpValue ?outValue WHERE {
	              ?activity prov:used ?input .
		      ?output   prov:wasGeneratedBy ?activity .
		      ?activity a ?moduleClass .
		      ?moduleClass rdfs:subClassOf prov:Activity .
		      ?input a ?inputClass .
		      ?inputClass rdfs:subClassOf prov:Entity .
		      ?output a ?outputClass .
		      ?outputClass rdfs:subClassOf prov:Entity .
		      ?input d2s:value ?inpValue .
		      ?output d2s:value ?outValue .
                      MINUS { ?input prov:wasGeneratedBy ?a } .
		      } ORDER BY ?moduleClass ?activity ?output
	   """
    result = g.query(qAIO)
    return result


def queryAllOutputs(g):
    qData = """
    	    SELECT ?output ?outputLabel ?value ?module ?moduleInstance WHERE {
			?moduleInstance	a ?moduleClass 
			?moduleClass rdfs:subClassOf prov:Activity .
			?moduleClass rdfs:label ?module .
			?output a ?outputClass .
			?outputClass rdfs:subClassOf prov:Entity .
	    		?output prov:wasGeneratedBy ?moduleInstance . 
			?output d2s:value ?value .

                        ?outputClass rdfs:label ?outputLabel  .
                        MINUS { ?a prov:used ?output } .

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

def queryLastActivities(g):
    qLastAct = """
               SELECT DISTINCT ?activity ?actLabel WHERE {
		   ?out prov:wasGeneratedBy ?activity .
                   ?activity a ?activityClass .
		   ?activityClass rdfs:subClassOf prov:Activity .
                   ?activityClass rdfs:label ?actLabel .
		   MINUS { ?a prov:used ?out } .
               } ORDER BY ?activity """
    result = g.query(qLastAct)
    return result

def queryInputs(g, activity):
    qInput= """
    	    SELECT ?input ?label ?value ?actLabel ?isAggregator WHERE {
	           <%s> prov:used ?input . 			
	           <%s> a ?actClass .
		   ?actClass rdfs:subClassOf prov:Activity .
                   ?actClass rdfs:label ?actLabel .
                   ?input a ?inputClass .
		   ?inputClass rdfs:subClassOf prov:Entity .
                   ?inputClass rdfs:label ?label .
                   ?input d2s:value ?value .
                   OPTIONAL { ?input a ?isAggregator . FILTER (?isAggregator = d2s:Aggregator ) }
		} """ % (activity,activity)
    result = g.query(qInput)
    return result

def queryOutputs(g, activity):
    qInput= """
    	    SELECT ?output ?label ?value WHERE {
	           ?output prov:wasGeneratedBy <%s> . 				
		   ?output a ?outputClass .
		   ?outputClass rdfs:subClassOf prov:Entity .
                   ?outputClass rdfs:label ?label .
                   ?output d2s:value ?value .
		} ORDER BY ?output ?value """ % activity
    result = g.query(qInput)
    return result

def queryParent(g, activity):
    qParent = """
              SELECT DISTINCT ?parent ?parentLabel WHERE {
                   ?link prov:wasGeneratedBy ?parent .
                   ?parent d2s:instanceOf ?parentClass .
                   ?parent rdfs:label ?parentLabel .
                   <%s> prov:used ?link .
              }
              """ % activity
    result = g.query(qParent)
    return result
 
def getInput(g, activity, depth):
    records = queryInputs(g, activity)
    result = {} 

    for rec in  records:
	cur = {}
        moduleName = rec["actLabel"].decode()
        inputName = rec["label"].decode()
        value = str(rec["value"].decode())
        if(rec["isAggregator"]):
	   inputName += ".Agg"
        cur[moduleName+"."+inputName]= value
        result.update(cur)
    return result

def getOutput(g, activity):
    records = queryOutputs(g, activity)
    result = {} 
    for rec in  records:
	cur = {}
        cur[rec["label"].decode()] = str(rec["value"].decode())
        result.update(cur)
    return result

def getParents(g, activity):
    records = queryParent(g, activity)
    parents = []
    parentLabels = set() 
    for rec in records:
         parents.append(rec["parent"])
         parentLabels.add(rec["parentLabel"].decode())
    parentLabels = sorted(parentLabels)
    return parents, " ".join(parentLabels)
 
def recGetInput(g, activity, depth):
    result = getInput(g, activity, depth) 
    parents, parentLabels = getParents(g, activity)
    parentKey = "parent"+str(depth)
    result[parentKey] = parentLabels
    for p in parents:
        result.update(recGetInput(g, p, depth+1) )
    return result
	
def queryAllInputOutput(g):
    # Start from last activities whose outputs are not used by other activities 
    activities = queryLastActivities(g)
    result = {}
    result["headers"] = ["Instance", "Inputs", "Outputs "] 
    result["fields"] = ["instance", "inputs", "outputs"] 
    result["shorten"] = [True,False,False] 
    result["records"] = []


    for act in activities :
        inputs = recGetInput(g, act["activity"], 0)
        outputs = getOutput(g,act["activity"])
        curRecord = {}
        curRecord["inputs"] = str(inputs).replace("'"," ")
        curRecord["outputs"] = str(outputs).replace("'"," ")
        curRecord["instance"] = shortenURL("%s" % act["activity"])
        result["records"].append( curRecord )

    return result    

# Extracting inputs and outputs of activities
def createPandaDataDict(g):
    # Start from last activities whose outputs are not used by other activities 
    result = {}
    order  = {} 
    inputSet = {} 
    output = {}
    activities = queryLastActivities(g)
    for act in activities :
        curModule = act["actLabel"]
        activity = act["activity"]

        #Get all parent input recursively for this activity 
        inputs = recGetInput(g, activity, 0)
        outputs = getOutput(g, activity)
        
        if curModule not in result:
           result[curModule] = [] 
           order[curModule] = outputs.keys()
           output[curModule] = [x.decode() for x in outputs.keys()]
           inputSet[curModule] = set()

        for i in inputs.keys():
           inputSet[curModule].add(i.decode())           

        curRecord = {}
        curRecord.update(outputs)
        curRecord.update(inputs)

        result[curModule].append( curRecord )
    
    for module in result :
        for v in inputSet[module]:
           order[module].append(v)

    return result,order, output    




###########################################################################
# Ipython Notebook cells generation code 
###########################################################################

def getPandaHeader():
    header = """
import datetime

import pandas as pd
import pandas.io.data
from pandas import Series, DataFrame
from pandas.tools.pivot import pivot_table 
import matplotlib.pyplot as plt
import matplotlib as mpl
"""
    return header


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

def createPandaCode(pandaDict, pandaOrder, outputList):
    result = """ %s
df = DataFrame(%s)
df = df[%s]

#Ignore constant column
df = df.loc[:, (df != df.ix[0]).any()] 
df
""" % (getPandaHeader(), str(pandaDict), str(pandaOrder))   

    if len(outputList) == 0 : 
	return result 

    result += """
outputList = %s
df[outputList] = df[outputList].astype(float)

cons=[]
for x in df.columns:
    if x.endswith(".Agg") or  x.find('parent') >=0 or not x.find(".") >=0  :
       continue
    cons.append(x)

df['Aggregator'] = df[cons[0]].astype(str)
for x in cons[1:]:
    df['Aggregator'] += df[x].astype(str)

aggIndexMap = {}
for x in df['Aggregator']: 
    if x not in aggIndexMap :
       aggIndexMap[x] = len(aggIndexMap) 

for i in range(len(df['Aggregator'])):
    df['Aggregator'][i] = aggIndexMap[df['Aggregator'][i]]

pivotRows = ['Aggregator']
if 'parent0' in df.columns:
   pivotRows.append('parent0')

pt = pivot_table(df, rows= pivotRows)
pt

""" % str(outputList)   
    return result
    
def createPandaTable(records, headers, fields, shorten):
    dataList = "["
    for row in records:
	dataList +=  "["
	for i in range(len(fields)):
            if(shorten[i]):
	       dataList += "'" + shortenURL(row[fields[i]]) +"',"
            else:
	       dataList += "'" + str(row[fields[i]]) +"',"
	dataList +=  "],"

    dataList += "]"

    return "from pandas import DataFrame\ndf = DataFrame("+dataList+", columns="+str(headers)+")\ndf"

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
	       result += "'" + str(row[fields[i]]) +"',"
	result +=  "],"

    result += "]"

    result += """
make_table(data)
apply_theme('basic') """
    return result

def createProvoVizCode(filename):
    f=open(filename,'r')
    data = f.read()

    code = """
from IPython.display import HTML
import requests
import hashlib 
data = \"\"\" """ +data + """\"\"\" 
    
digest = hashlib.md5(data).hexdigest()
graph_uri = "http://data2semantics.org/platform/{}".format(digest)
    
payload = {'graph_uri': graph_uri, 'data' : data }
response = requests.post("http://localhost:5000/service",data=payload)
    
html_filename = '{}_provoviz.html'.format(digest)
html_file = open(html_filename,'w')
html_file.write(response.text)
html_file.close()
    
iframe = "<iframe width='100%' height='450px' src='http://localhost:8000/{}'></iframe>".format(html_filename)
    
HTML(iframe)""" 
    return code


def shortenURL(longURL):
    try :
       result = longURL.split("/")[-1]
    except:
       return ""
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
	curInstance = str(row["output"])
	curOutput  = str(curInstance.split("/")[-1])
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

def hideOnce():
    return """<script type="text/javascript"> $('div.input').hide(); </script>"""

def getHideAllInputCell():
    hideMD = """
<script type="text/javascript">
function toggleInputCode(){
   if($('div.input').css('display') == 'none'){
       $('div.input').show();
   } else 
       $('div.input').hide();
   }
</script>
<button id="showInputCode"  title="Toggle input code" onclick="toggleInputCode()">View/Hide Input Code</button>
</script>
"""
    return hideMD

###########################################################################
# Main code 
###########################################################################

#Prepare prefix and load the main input file
def loadProvGraph (inputFileName):

    provgraph = rdflib.Graph()
    
    rdf  = rdflib.Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    rdfs = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")
    d2s  = rdflib.Namespace("http://prov.data2semantics.org/vocab/")
    prov = rdflib.Namespace("http://www.w3.org/ns/prov#")
    
    provgraph.bind('rdf',  rdf)
    provgraph.bind('rdfs', rdfs)
    provgraph.bind('d2s',  d2s)
    provgraph.bind('prov', prov)
    
    # Need to give this as parameter instead of hard coding, will do this later.
    provgraph.parse(inputFileName, format="n3")

    return provgraph
    
#==========================================================================
# Main Notebook
#==========================================================================

def createMainNoteBook(inputfile, provgraph, outputPrefix):
    cells = []
    cells.append(createHeaderCell("Overview Report",1))
    cells.append(createHeaderCell("Software",2))
    
    planData = queryThePlan(provgraph)

    for planRow in planData:
       cells.append(createMarkdownCell(getPlanInfo(planRow), 3))
       cells.append(createHeaderCell("Libraries",3))
       dependencyData = queryDependencies(provgraph)
       dependencyCode = createIPYTable(dependencyData, ["GroupID",  "ArtifactID", "Version"], ["groupID","artifactID", "versionID"], [ False, False,False])
       cells.append(createCodeCell(dependencyCode));

    
    
    cells.append(createHeaderCell("Modules",2))
    modules = queryAllModules(provgraph)
    moduleCode = createIPYTable(modules, ["Module",  "Instances"], ["module","moduleInstance"], [ False, False])
    cells.append(createCodeCell(moduleCode));
    
    cells.append(createHeaderCell("Datasets",2))
    dataSets = queryAllDataset(provgraph)
    datasetCode = createIPYTable(dataSets, ["Module", "Dataset","Value"], ["activityLabel","inputLabel", "value"], [False, False, False])
    cells.append(createCodeCell(datasetCode))
    
#    cells.append(createHeaderCell("Results",2))
#    outputSets = queryAllOutputs(provgraph)
#    outputCode = createPandaTable(outputSets, ["Module ","Output", "Value"], ["module", "outputLabel", "value"], [False, False, False])
#    cells.append(createCodeCell(outputCode))
#    
    cells.append(createHeaderCell("Provenance Visualization",1))
    cells.append(createCodeCell(createProvoVizCode(inputfile)))

    cells.append(createHeaderCell("Details",1))
    cells.append(createMarkdownCell("[Detailed Information]("+outputPrefix+"-detail.ipynb)"))
    
    #Hide all inputs
    cells.append(createMarkdownCell(hideOnce()))
    cells.append(createMarkdownCell(getHideAllInputCell()))
    
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
    
    main = open(outputPrefix+"-main.ipynb","w")
    main.write(json.dumps(result))
    main.close()
    
#==========================================================================
#Detailed Notebook
#==========================================================================

def createDetailedNotebook(provgraph, outputPrefix):
    cells = []

    cells.append(createMarkdownCell("[Main Notebook]("+outputPrefix+"-main.ipynb)"))

    cells.append(createHeaderCell("Activity Runtimes ",2))
    activities = queryAllActivities(provgraph)
    activityCode = createIPYTable(activities, ["Activity", "Start", "Stop "], ["activity","startTime","endTime"], [True, False, False])
    cells.append(createCodeCell(activityCode));
    
#   cells.append(createHeaderCell("Detailed Activities",1))

#    cells.append(createHeaderCell("Activities input output", 2))
#
#    activitiesRec = queryActivityInputOutput(provgraph)
#    activityIpyTableCode = createIPYTable(activitiesRec, ["Activity", "Input", "Value", "Output", "Value"], ["activity","input","inpValue", "output", "outValue"], [False, True, True,True,True])
#    cells.append(createCodeCell(activityIpyTableCode, collapsed="true"));
#    cells.append(createHeaderCell("Activities Code ", 2))
#    
#    activitiesCells = createActivityCodes(activitiesRec)
#    
#    for actCell in activitiesCells:
#        actCell = "#This will be pseudo code which can be used to call Ducktape for verification\n"+actCell
#        cells.append(createCodeCell(actCell))
    
    
#    cells.append(createHeaderCell("All Input  Outputs",2))
#
#    allIOTable = queryAllInputOutput(provgraph)
#    allIOTableCode = createIPYTable(allIOTable["records"], allIOTable["headers"], allIOTable["fields"], allIOTable["shorten"])
#    cells.append(createCodeCell(allIOTableCode, collapsed="True"));
#
    cells.append(createMarkdownCell(hideOnce()))
    cells.append(createMarkdownCell(getHideAllInputCell()))
    cells.append(createHeaderCell("Experiment Results Per Modules",2))
    pandaDict, pandaOrder, pandaOutput = createPandaDataDict(provgraph)

    for module in pandaDict :
        curDict = pandaDict[module]
        curOrder = pandaOrder[module]
        curOutput = pandaOutput[module]
	
        cells.append(createHeaderCell("Result for module " + str(module),3))
        pandaCode = createPandaCode(curDict, curOrder, curOutput)
        cells.append(createCodeCell(pandaCode))
 


#    cells.append(createHeaderCell("Output Plots ",2))
#    outputSets = queryAllOutputs(provgraph)
#    outputValues = groupByOutputType(outputSets)
#    
#    for curModule in outputValues:
#        cells.append(createHeaderCell(str(curModule),2))
#        cells.append(createHeaderCell("Outputs",3))
#        cells.append(createCodeCell(plotModuleOutputCode(outputValues[curModule])))
#   
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
   

    
    detail = open(outputPrefix+"-detail.ipynb","w")
    detail.write(json.dumps(result))
    detail.close()
    
def main(argv):
    inputfile = ''
    outputPrefix= 'nb'
    try:
       opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
       print 'prov2nb.py -i <inputfile> -o <outputPrefix>'
       sys.exit(2)

    for opt, arg in opts:
       if opt == '-h':
          print 'prov2nb.py -i <inputfile> -o <outputPrefix>'
          sys.exit()
       elif opt in ("-i", "--ifile"):
          inputfile = arg
       elif opt in ("-o", "--ofile"):
          outputPrefix = arg

    if inputfile == '':
       print 'prov2nb.py -i <inputfile> -o <outputPrefix>'
       sys.exit(2)

    provgraph = loadProvGraph(inputfile)
    createMainNoteBook(inputfile, provgraph, outputPrefix)
    createDetailedNotebook(provgraph, outputPrefix)


if __name__ == "__main__":
   main(sys.argv[1:])
    
