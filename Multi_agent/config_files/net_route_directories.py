# Define a list of tuples where each tuple contains the name, net file, and route file paths
file_locations = [
    ("2x2grid", "./nets/2x2grid/2x2.net.xml", "./nets/2x2grid/2x2.rou.xml"),
    ("3x3grid", "./nets/3x3grid/3x3.net.xml", "../nets/3x3grid/3x3.rou.xml"),
    ("4x4-Lucas", "./nets/4x4-Lucas/4x4.net.xml", "./nets/4x4-Lucas/4x4c1.rou.xml"),
    ("big-intersection", "./nets/big-intersection/big-intersection.net.xml", "./nets/big-intersection/routes.rou.xml"),
    ("ingolstadt1", "./nets/ingolstadt1/ingolstadt1.net.xml", "./nets/ingolstadt1/ingolstadt1.rou.xml"),
    ("ingolstadt7", "./nets/ingolstadt7/ingolstadt7.net.xml", "./nets/ingolstadt7/ingolstadt7.rou.xml"),
    ("ingolstadt21", "./nets/ingolstadt21/ingolstadt21.net.xml", "./nets/ingolstadt21/ingolstadt21.rou.xml"),
    ("beyersRand", "./nets/beyers/beyers.net.xml", "./nets/beyers/beyers_rand.rou.xml"),
    ("beyers", "./nets/beyers/beyers.net.xml", "./nets/beyers/beyers.rou.xml"),
    ("cologne1", "./nets/cologne1/cologne1.net.xml", "./nets/cologne1/cologne1.rou.xml"),
    ("cologne3", "./nets/cologne3/cologne3.net.xml", "./nets/cologne3/cologne3.rou.xml"),
    ("cologne8", "./nets/cologne8/cologne8.net.xml", "./nets/cologne8/cologne8.rou.xml"),
    ("simple", "./nets/simple/simple.net.xml", "./nets/simple/simple.rou.xml")
]

# Function to retrieve the file locations by name
def get_file_locations(name):
    for entry in file_locations:
        if entry[0] == name:
            return {"net": entry[1], "route": entry[2]}

