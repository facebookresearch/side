
class TaskDataset:
    def __init__(self,
                 file,
                 task_family,
                 name,
                 question_transform_type,
                 ):
        self.file = file
        self.task_family = task_family
        self.name = name
        self.question_transform_type = question_transform_type

    def validate_datapoint(self, datapoint, logger):
        raise NotImplementedError

    def transform_query(self, datapoint, question_transform_type):
        return datapoint


class WaferCCNetTaskDataset(TaskDataset):

    def validate_datapoint(self, datapoint, logger):

        # input is a string
        if not isinstance(datapoint["input"], str):
            if logger:
                logger.warning(
                    "[{}] input is not a string {}".format(
                        datapoint["id"], datapoint["input"]
                    )
                )
            return False

        # output is not empty
        if "output" in datapoint:
            if len(datapoint["output"]) == 0:
                if logger:
                    logger.warning("[{}] empty output".format(datapoint["id"]))
                return False

            for output in datapoint["output"]:
                # answer is a string
                if "answer" in output:
                    if not isinstance(output["answer"], str):
                        if logger:
                            logger.warning(
                                "[{}] answer is not a string {}".format(
                                    datapoint["id"], output["answer"]
                                )
                            )
                        return False

                # provenance is not empty
                if len(output["provenance"]) == 0:
                    if logger:
                        logger.warning("[{}] empty provenance".format(datapoint["id"]))
                    return False

                if "provenance" in output:
                    for provenance in output["provenance"]:
                        # title is provided
                        if "url" not in provenance or not isinstance(provenance["url"], str):
                            if logger:
                                logger.warning("datapoint with malformed or missing url")
                            return False

        return True
