def evaluate_places(filepath, predicted_places):
  """ Computes percent of correctly predicted birth places.

  Arguments:
    filepath: path to a file with our name, birth place data.
    predicted_places: a list of strings representing the
        predicted birth place of each person.

  Returns: (total, correct), floats
  """
  print("Only comparing with 'London'")
  with open(filepath, encoding='utf-8') as fin:
    lines = [x.strip().split('\t') for x in fin]
    if len(lines[0]) == 1:
      print('No gold birth places provided; returning (0,0)')
      return (0,0)
    true_places = [x[1] for x in lines] # for each line: get the true answer at the 2nd place x[1]
    total = len(true_places)
    assert total == len(predicted_places)
    predicted_places = ['London' for place in predicted_places]
    correct = len(list(filter(lambda x: x[0] == x[1],
      zip(true_places, predicted_places))))
    return (float(total),float(correct))