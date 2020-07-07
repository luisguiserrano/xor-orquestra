import json
import matplotlib

with open('./xor-results.json') as f:
    data = json.load(f)

predictions = data['xor-orq-vpwqb-2061171285']['result']['predictions']

print(predictions)