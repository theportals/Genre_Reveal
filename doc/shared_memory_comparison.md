# Serial vs Parallel Shared Memory CPU vs Parallel Shared Memory GPU
All tests were ran with k=5, x=danceability, y=speechiness, z=acousticness, and verified to be consistent with serial output. Each implementation took 17 epochs to converge.

Time to read source csv was ignored.

Hardware used: Ryzen 5 3600, and an RTX 2060.

## Serial
|                    | Serial Implementation |
|--------------------|-----------------------|
| Time to converge   | 3402ms                |

## Shared CPU
12 threads was included since that's the max logical cores on the ryzen 5 3600

| Threads            | 1      | 2      | 4     | 8     | 12    | 16    | 32    |
|--------------------|--------|--------|-------|-------|-------|-------|-------|
| Time to converge   | 1764ms | 1156ms | 897ms | 787ms | 717ms | 780ms | 748ms |

## Shared GPU
1024 is the max block size on the RTX 2060, as well as most modern NVIDIA cards.

| Block size         | 1     | 2     | 4    | 8    | 16   | 32   | 64   | 128  | 256  | 512  | 1024 |
|--------------------|-------|-------|------|------|------|------|------|------|------|------|------|
| Time to converge   | 347ms | 171ms | 88ms | 55ms | 53ms | 50ms | 51ms | 50ms | 50ms | 62ms | 59ms |
