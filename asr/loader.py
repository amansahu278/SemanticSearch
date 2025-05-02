from utils.utils import str_to_datetime


def parse_asr(file_path, word=False):
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i] == "\n":
                timestamp = lines[i + 1]
                start_time = str_to_datetime(timestamp.strip().split("-->")[0].strip())
                end_time = str_to_datetime(timestamp.strip().split("-->")[1].strip())

                text = lines[i + 2]
                if word:
                    for word in text.split():
                        if len(word.strip()) > 0:
                            data.append(
                                {
                                    "start_time": start_time,
                                    "end_time": end_time,
                                    "text": word.strip(),
                                }
                            )
                else:
                    data.append(
                        {
                            "start_time": start_time,
                            "end_time": end_time,
                            "text": text.strip(),
                        }
                    )
    return data


def load_asr(file_path, word=False):
    return parse_asr(file_path, word)
