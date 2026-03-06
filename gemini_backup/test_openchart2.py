from openchart import NSEData
nse = NSEData()
print(nse.search("RELIANCE", segment="EQ"))
print(nse.search("RELIANCE", segment="IDX"))
print(nse.search("RELIANCE-EQ"))
