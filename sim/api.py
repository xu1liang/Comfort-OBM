import os
import requests
import json
from .log import *
def DumpSampleJson(jsonData, fileName, overwrite=False):
    abspathFileName = os.path.abspath(fileName)
    dirname = os.path.dirname(abspathFileName)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not os.path.exists(abspathFileName) or overwrite:
        with open(abspathFileName, "w") as f:
            json.dump(jsonData, f, indent=4)


class SimTaskResult(LoggerImpl):
    def __init__(self, taskId):
        self.taskId = taskId
        self.header = {'Authorization': '6065bffcdeea8f0e4cbe636042868310'}
        self.collectorDirName = __class__.__name__

    def GetResult(self, resultHandler=None):
        if resultHandler is None:
            resultFileName = f"{self.collectorDirName}/{self.taskId}.json"
        else:
            resultFileName = f"{self.collectorDirName}/{self.taskId}.resultHandler.json"
        checkersResultMap = {}
        try:
            if os.path.exists(resultFileName):
                with open(resultFileName, "r") as f:
                    checkersResultMap = json.load(f)
                    self.Info(f"Loaded MetricsResultMap from {resultFileName}")
        except Exception as e:
            logging.error(f"{type(e).__name__}: {e}")

        if not checkersResultMap:
            checkersResultMap = self.QueryCheckerResultMap(resultHandler)
            DumpSampleJson(checkersResultMap, resultFileName, overwrite=True)
            self.Info(f"Dumped checkersResultMap to {resultFileName}")

        return checkersResultMap

    def QueryCheckerResultMap(self, resultHandler=None):
        def RequestSimCheckerResult(checkerResultUrl):
            url = f"https://simulation.momenta.works/portal/{checkerResultUrl}"
            return requests.get(url, headers=self.header).json()
        resultMap = {}
        if self.IsSimTaskCompleted():
            simEventListResult = self.RequestSimEventListResult()
            for eventResultMeta in simEventListResult:
                DumpSampleJson(eventResultMeta, f"{self.collectorDirName}/simEventListSample.json")

                itemId = eventResultMeta["item_id"]
                scenarioName = eventResultMeta["scenario_name"]
                checkerResultUrl = eventResultMeta["checker_result_url"]
                if checkerResultUrl:
                    try:
                        checkerResults = RequestSimCheckerResult(checkerResultUrl)
                    except Exception as e:
                        self.Error(f"{type(e).__name__}: {e}")
                        checkerResults = None
                if not checkerResults:
                    self.Warning(f"Failed to get checker results for {scenarioName}")
                    global interactiveFlag
                    if interactiveFlag:
                        yn = input("Continue? (y/n/a)")
                        if yn == "n":
                            raise e
                        elif yn == "a":
                            interactiveFlag = False
                    continue
                eventResultMeta["checker_result"] = checkerResults
                if callable(resultHandler):
                    checkerResults = resultHandler(eventResultMeta)
                resultMap[f"{scenarioName}.{itemId}"] = eventResultMeta
        else:
            raise RuntimeError(f"Task {self.taskId} is not completed")

        return resultMap

    def IsSimTaskCompleted(self):
        result = False
        url = f"https://simulation.momenta.works/portal/api/v1/task/{self.taskId}/info"
        try:
            print(f"Requesting {url}")
            taskInfo = self.GetTaskInfo()
            taskStatus = taskInfo["status"]
            taskname = taskInfo["name"]
            self.Info(f"Task {self.taskId} {taskname} status: {taskStatus}")
            result = taskStatus == "COMPLETED"
        except Exception as e:
            self.Error(f"{type(e).__name__}: {e}")
        return result

    def GetTaskInfo(self):
        url = f"https://simulation.momenta.works/portal/api/v1/task/{self.taskId}/info"
        try:
            response = requests.get(url, headers=self.header)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            self.Error(f"{type(e).__name__}: {e}")
        return {}

    def RequestSimEventListResult(self):
        result = []
        page = 0
        size = 100
        url = f"https://simulation.momenta.works/portal/api/v1/task/{self.taskId}/item_list?page={page}&size={size}"
        sum = 0
        total = 0
        while True:
            try:
                response = requests.get(url, headers=self.header)
                if response.status_code == 200:
                    responseJson = response.json()
                    total = responseJson["total"]
                    result.extend(responseJson["events"])
                    sum += size
                    if sum < total:
                        page += 1
                        url = f"https://simulation.momenta.works/portal/api/v1/task/{self.taskId}/item_list?page={page}&size={size}"
                    else:
                        return result
                else:
                    self.error(f"RequestSimEventListResult failed: {response.status_code} {response.text}")
                    self.error(f"RequestSimEventListResult url: {url}")
                    break
            except Exception as e:
                self.error(f"{type(e).__name__}: {e}")
                break
        self.Info(f"Get {len(result)} event set results")
        return result

def ExtractFeatureMetrics(task_id_list, checker):
    def _getFeaturesFromResult(result, checker):
        try:
            metrics_analyser = result["checker_result"]["checkers_result"][checker]["data"]["metrics_analyser"]
            return metrics_analyser, result["scenario_name"]
        except:
            return None, None
    def _getFeatures(taskId, checker):
        features = []
        names = []
        allResults = SimTaskResult(taskId).GetResult()
        LoggerImpl().Info(f"Total len: {len(allResults)}")
        for _, result in allResults.items():
            feature, name = _getFeaturesFromResult(result, checker)
            if (feature is not None) and (name is not None):
                features.append(feature)
                names.append(name)
        return features, names
    features_list = []
    names_list = []
    for task_id in task_id_list:
        feature, name = _getFeatures(task_id, checker)
        LoggerImpl().Info(f"Success len: {len(feature)}")
        features_list.append(feature)
        names_list.append(name)
    return features_list, names_list