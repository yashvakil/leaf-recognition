PRAGMA foreign_keys = OFF;

DROP TABLE IF EXISTS 'TreeData';
DROP TABLE IF EXISTS 'Processed';

PRAGMA foreign_keys = ON;

CREATE TABLE 'TreeData'(
    'TreeID' INTEGER PRIMARY KEY AUTOINCREMENT,
    'ScientificName' TEXT NOT NULL,
    'CommonName' TEXT NOT NULL,
    'Path' TEXT NOT NULL
);

CREATE TABLE 'Processed'(
    'TreeID' INTEGER NOT NULL,
    'LeafID' TEXT NOT NULL,
    'Length' REAL NOT NULL,
    'Width' REAL NOT NULL,
    'Area' REAL NOT NULL,
    'Perimeter' REAL NOT NULL,
    'AspectRatio' REAL NOT NULL,
    'FormFactor' REAL NOT NULL,
    'Rectangularity' REAL NOT NULL,
    'Hu' TEXT NOT NULL,
    'Hist' TEXT NOT NULL,
    PRIMARY KEY ('TreeId', 'LeafID'),
    FOREIGN KEY('TreeId') REFERENCES TreeData('TreeID')
);