CREATE TABLE Administrator(
    Email VARCHAR2(20) PRIMARY KEY,
    Password VARCHAR2(69) NOT NULL
);

CREATE TABLE Attendee(
    Account_ID VARCHAR2(20) PRIMARY KEY,
    Password VARCHAR2(69) NOT NULL,
    First_name VARCHAR2(20) NOT NULL,
    Last_name VARCHAR2(20) NOT NULL,
    Organization VARCHAR2(5) NOT NULL, /* SPEED PolyU HKCC */
    Attendee_type VARCHAR2(7) NOT NULL,
    Address VARCHAR2(50) NOT NULL,
    Mobile_num CHAR(8) NOT NULL
);

CREATE TABLE Banquet(
    BIN CHAR(10) PRIMARY KEY,
    Banquet_name VARCHAR2(20) NOT NULL,
    Banquet_date DATE NOT NULL,
    Banquet_time CHAR(8) NOT NULL,
    Location VARCHAR2(50) NOT NULL,
    Address VARCHAR2(50) NOT NULL,
    First_name VARCHAR2(20) NOT NULL,
    Last_name VARCHAR2(20) NOT NULL,
    Availability NUMBER(1) NOT NULL,
    Quota NUMBER(3) NOT NULL
);

CREATE TABLE Meal(
    Dish_name VARCHAR2(20) NOT NULL,
    BIN CHAR(10) NOT NULL,
    Cuisine VARCHAR2(20) NOT NULL,
    Price NUMBER NOT NULL,
    Type VARCHAR(20) NOT NULL,
    PRIMARY KEY (BIN, Dish_name),
    FOREIGN KEY (BIN) REFERENCES Banquet(BIN) ON DELETE CASCADE
);

CREATE TABLE Registry(
    Account_ID VARCHAR2(20) NOT NULL,
    BIN CHAR(10) NOT NULL,
    Dish_name VARCHAR2(20) NOT NULL,
    Drink VARCHAR2(20) NOT NULL,
    Seat CHAR(3) NOT NULL,
    Attendance NUMBER(1) NOT NULL,
    Remarks VARCHAR2(100) NOT NULL,
    PRIMARY KEY (Account_ID, BIN, Dish_name),
    FOREIGN KEY (BIN) REFERENCES Banquet(BIN) ON DELETE CASCADE,
    FOREIGN KEY (BIN, Dish_name) REFERENCES Meal(BIN, Dish_name) ON DELETE CASCADE,
    FOREIGN KEY (Account_ID) REFERENCES Attendee(Account_ID) ON DELETE CASCADE
);

INSERT INTO Administrator (Email, Password) VALUES ('admin1@mail.com', 'nPb8zO4cL6v/Ps9qTSRzrQ==:z3qJAeL9hGVeuj5rmmOwDmcUVuDC4t+mjlUANPgJq3s=');
INSERT INTO Administrator (Email, Password) VALUES ('admin2@mail.com', 'LonZ2V6VFc8gLSQHjkTtEQ==:f51X8QtpVE7bNpfRYYJqs655YgrhHTNpAB3KB3OuIPE=');
INSERT INTO Administrator (Email, Password) VALUES ('admin3@mail.com', 'eSufwd5sWDXKoG2qppWo4A==:dmcUT6lt2UiHYtljUAycmLxgFxrjugryzF2V0PwP4TI=');

SELECT * FROM Administrator;
SELECT Password FROM Administrator WHERE Email = 'admin1@mail.com';