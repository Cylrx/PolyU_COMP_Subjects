import math
import random
import datetime

COMMANDS = [
    "addParking -member_{member} {date} {time} {duration} {essentials};",
    "addReservation -member_{member} {date} {time} {duration} {essentials};",
    "addEvent -member_{member} {date} {time} {duration} {essentials};",
    "bookEssentials -member_{member} {date} {time} {duration} {essential};"
]

MEMBERS = ["A", "B", "C", "D", "E"]

ESSENTIAL_PAIRS = [
    "battery cable",
    "locker umbrella",
    "InflationService valetPark"
]

ESSENTIALS = ["battery", "cable", "locker", "umbrella", "InflationService", "valetPark"]


# Multimodal Normal Distribution
MU_1 = [10 + 24 * k for k in range(7)]
MU_2 = [14 + 24 * k for k in range(7)]
SIGMA = 1
WEIGHT_1 = WEIGHT_2 = 0.5

# Exponential Distribution
LAMBDA_NIGHT = 1 / 10
LAMBDA_DAY = 1 / 2

# Date Range
START_DATE = datetime.datetime(2025, 5, 10)
END_DATE = datetime.datetime(2025, 5, 16)
TOTAL_HOURS = int((END_DATE - START_DATE).total_seconds() / 3600)

def generate_bimodal_time():
    while True:
        if random.random() < WEIGHT_1:
            mu = random.choice(MU_1)
        else:
            mu = random.choice(MU_2)

        time = random.gauss(mu, SIGMA)

        if 0 <= time < TOTAL_HOURS:
            return int(time)

def generate_exponential_duration(start_hour):
    tau = start_hour % 24
    if 0 <= tau < 4 or 20 <= tau < 24:
        lam = LAMBDA_NIGHT
    else:
        lam = LAMBDA_DAY

    duration = math.ceil(random.expovariate(lam))
    return max(1, min(duration, 14))

def generate_essentials():
    num_essentials = random.randint(0, 3)
    selected_essentials = random.sample(ESSENTIALS, num_essentials)
    return " ".join(selected_essentials)

def generate_command():
    command_template = random.choice(COMMANDS)
    member = random.choice(MEMBERS)

    start_hour = generate_bimodal_time()
    start_date = START_DATE + datetime.timedelta(hours=start_hour)
    date = start_date.strftime("%Y-%m-%d")
    time = start_date.strftime("%H:%M")

    duration = str(generate_exponential_duration(start_hour)) + ".0"

    if "addParking" in command_template or "addReservation" in command_template or "addEvent" in command_template:
        essentials = random.choice(ESSENTIAL_PAIRS)
    else:  # bookEssentials
        essentials = generate_essentials()

    command = command_template.format(
        member=member,
        date=date,
        time=time,
        duration=duration,
        essentials=essentials,
        essential=essentials
    )

    return command.replace(" ;", ";")

def generate_batch_file(n, output_file="batch-dp.dat"):
    with open(output_file, "w") as f:
        for _ in range(n):
            command = generate_command()
            f.write(command + "\n")
    print(f"Batch file with {n} commands generated: {output_file}")

generate_batch_file(2000)