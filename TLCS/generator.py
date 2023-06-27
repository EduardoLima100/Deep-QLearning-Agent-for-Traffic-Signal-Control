import random

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        random.seed(seed)  # make tests reproducible

        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print(f"""<routes>
    <!-- VTypes -->
    <vType id="car" length="5.00" minGap="2.50" maxSpeed="25.00" guiShape="passenger" accel="1.0" decel="4.5" sigma="0.5"/>
    
    <!-- Routes -->
    <route id="north_route" edges="north_in south_out"/>
    <route id="south_route" edges="south_in north_out"/>
    <route id="east_route" edges="east_in west_out"/>
    <route id="west_route" edges="west_in east_out"/>
    
    <!-- Vehicles sorted by depart -->
    <flow id="north0" type="car" begin="0.00" route="north_route" end="300.00" probability="{random.randint(2,10)/100}"/>
    <flow id="south0" type="car" begin="0.00" route="south_route" end="300.00" probability="{random.randint(2,10)/100}"/>
    <flow id="east0" type="car" begin="0.00" route="east_route" end="300.00" probability="{random.randint(2,10)/100}"/>
    <flow id="west0" type="car" begin="0.00" route="west_route" end="300.00" probability="{random.randint(2,10)/100}"/>

    <flow id="north1" type="car" begin="300.00" route="north_route" end="600.00" probability="{random.randint(2,10)/100}"/>
    <flow id="south1" type="car" begin="300.00" route="south_route" end="600.00" probability="{random.randint(2,10)/100}"/>
    <flow id="east1" type="car" begin="300.00" route="east_route" end="600.00" probability="{random.randint(2,10)/100}"/>
    <flow id="west1" type="car" begin="300.00" route="west_route" end="600.00" probability="{random.randint(2,10)/100}"/>

    <flow id="north2" type="car" begin="600.00" route="north_route" end="900.00" probability="{random.randint(2,10)/100}"/>
    <flow id="south2" type="car" begin="600.00" route="south_route" end="900.00" probability="{random.randint(2,10)/100}"/>
    <flow id="east2" type="car" begin="600.00" route="east_route" end="900.00" probability="{random.randint(2,10)/100}"/>
    <flow id="west2" type="car" begin="600.00" route="west_route" end="900.00" probability="{random.randint(2,10)/100}"/>

    <flow id="north3" type="car" begin="900.00" route="north_route" end="1200.00" probability="{random.randint(2,10)/100}"/>
    <flow id="south3" type="car" begin="900.00" route="south_route" end="1200.00" probability="{random.randint(2,10)/100}"/>
    <flow id="east3" type="car" begin="900.00" route="east_route" end="1200.00" probability="{random.randint(2,10)/100}"/>
    <flow id="west3" type="car" begin="900.00" route="west_route" end="1200.00" probability="{random.randint(2,10)/100}"/>

    <flow id="north4" type="car" begin="1200.00" route="north_route" end="1500.00" probability="{random.randint(2,10)/100}"/>
    <flow id="south4" type="car" begin="1200.00" route="south_route" end="1500.00" probability="{random.randint(2,10)/100}"/>
    <flow id="east4" type="car" begin="1200.00" route="east_route" end="1500.00" probability="{random.randint(2,10)/100}"/>
    <flow id="west4" type="car" begin="1200.00" route="west_route" end="1500.00" probability="{random.randint(2,10)/100}"/>

    <flow id="north5" type="car" begin="1500.00" route="north_route" end="1800.00" probability="{random.randint(2,10)/100}"/>
    <flow id="south5" type="car" begin="1500.00" route="south_route" end="1800.00" probability="{random.randint(2,10)/100}"/>
    <flow id="east5" type="car" begin="1500.00" route="east_route" end="1800.00" probability="{random.randint(2,10)/100}"/>
    <flow id="west5" type="car" begin="1500.00" route="west_route" end="1800.00" probability="{random.randint(2,10)/100}"/>
</routes>""", file=routes)

if __name__ == "__main__":
    TrafficGenerator(1000, 10).generate_routefile(random.randint(0, 1000))