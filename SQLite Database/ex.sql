BEGIN TRANSACTION;

ALTER TABLE 'Processed' RENAME TO tmp_table_name;

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

INSERT INTO Processed(TreeID, LeafID, Length, Width, Area, Perimeter, AspectRatio, FormFactor, Rectangularity, Hu, Hist) SELECT TreeID, LeafID, Width, Length, Area, Perimeter, AspectRatio, FormFactor, Rectangularity, Hu, Hist FROM tmp_table_name;

DROP TABLE tmp_table_name;

COMMIT;