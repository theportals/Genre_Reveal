# Distributed Memory CPU vs Distributed Memory GPU
All tests were ran with k=5, x=danceability, y=speechiness, z=acousticness, and verified to be consistent with serial output. Each implementation took 17 epochs to converge.

Time to read source csv was ignored.

Tests were run on the University of Utah's Notchpeak CHPC

## Distributed CPU

| Threads            | 1      | 2      | 4     | 8     | 16    | 32    |
|--------------------|--------|--------|-------|-------|-------|-------|
| Time to converge   | 2556ms | 1281ms | 640ms | 347ms | 180ms | 171ms |

## Distributed GPU
WIP