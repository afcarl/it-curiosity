import json
import pickle


class ObjectEncoder(json.JSONEncoder):
  def default(self, obj):
    if hasattr(obj, "to_json"):
      return self.default(obj.to_json())

    return obj

def dump_json(filename, results, overwrite=False):
  if overwrite:
    data = []
    try:
      with open(filename, "r") as f:
        data = json.load(f)
    except FileNotFoundError:
      pass
    finally:
      data.append(results)
  else:
    data = results

  with open(filename, "w") as f:
    json.dump(data, f, sort_keys=True, indent=4,
              separators=(',', ': '), cls=ObjectEncoder)

def dump_pickle(filename, results):
  with open(filename, "wb") as f:
    pickle.dump(results, f)

def dump_results(filename, results, file_format="json"):
  if file_format == "json":
    dump_json(filename, results)
  elif file_format == "pickle":
    dump_pickle(filename, results)
  else:
    raise ValueError("unexpected dump format")

def load_results(filename, file_format=None):
  if file_format == "json" or "json" in filename:
    with open(filename, "r") as f:
      results = json.load(f)
  elif file_format == "pickle" or "pickle" in filename:
    with open(filename, "rb") as f:
      results = pickle.load(f)
  else:
    raise ValueError("unexpected file format")

  return results
