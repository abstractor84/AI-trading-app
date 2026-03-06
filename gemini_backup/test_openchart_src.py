import inspect
from openchart import NSEData

print(inspect.getsource(NSEData.search))
print("---")
print(inspect.getsource(NSEData._fetch_historical))
print("---")
print(inspect.getsource(NSEData.historical))
