# Define a list of tuples where each tuple contains the name, net file, and route file paths
file_locations = [
    ("2x2grid", "./nets/2x2grid/2x2.net.xml", "./nets/2x2grid/2x2.rou.xml"),
    ("3x3grid", "./nets/3x3grid/3x3.net.xml", "../nets/3x3grid/3x3.rou.xml"),
    ("4x4-Lucas", "./nets/4x4-Lucas/4x4.net.xml", "./nets/4x4-Lucas/4x4c1.rou.xml"),
    ("big-intersection", "./nets/big-intersection/big-intersection.net.xml", "./nets/big-intersection/routes.rou.xml"),
    ("ingolstadt1", "./nets/ingolstadt1/ingolstadt1.net.xml", "./nets/ingolstadt1/ingolstadt1.rou.xml", "./nets/ingolstadt1/ingolstadt1.add.xml"),
    ("ingolstadt7", "./nets/ingolstadt7/ingolstadt7.net.xml", "./nets/ingolstadt7/ingolstadt7.rou.xml", "./nets/ingolstadt7/ingolstadt7.add.xml"),
    ("ingolstadt21", "./nets/ingolstadt21/ingolstadt21.net.xml", "./nets/ingolstadt21/ingolstadt21.rou.xml", "./nets/ingolstadt21/ingolstadt21.add.xml"),
    ("beyersRand", "./nets/beyers/beyers.net.xml", "./nets/beyers/beyers_rand.rou.xml", "./nets/beyers/beyers.add.xml"),
    ("beyers", "./nets/beyers/beyers.net.xml", "./nets/beyers/beyers.rou.xml", "./nets/beyers/beyers.add.xml"),
    ("cologne1", "./nets/cologne1/cologne1.net.xml", "./nets/cologne1/cologne1.rou.xml", "./nets/cologne1/cologne1.add.xml"),
    ("cologne3", "./nets/cologne3/cologne3.net.xml", "./nets/cologne3/cologne3.rou.xml", "./nets/cologne3/cologne3.add.xml"),
    ("cologne8", "./nets/cologne8/cologne8.net.xml", "./nets/cologne8/cologne8.rou.xml", "./nets/cologne8/cologne8.add.xml"),
    ("simple", "./nets/simple/simple.net.xml", "./nets/simple/simple.rou.xml")
]

# Function to retrieve the file locations by name
def get_file_locations(name):
    for entry in file_locations:
        if entry[0] == name:
            if len(entry) == 3:
                return {"net": entry[1], "route": entry[2]}
            elif len(entry) == 4:
                return {"net": entry[1], "route": entry[2], "additional": entry[3]}
    raise ValueError(f"{name} is not a valid file map name")

