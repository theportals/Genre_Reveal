# Distributed Memory CPU vs Distributed Memory GPU
All tests were ran with k=5, x=danceability, y=speechiness, z=acousticness, and verified to be consistent with serial output. Each implementation took 17 epochs to converge.

Time to read source csv was ignored.

Tests were run on the University of Utah's Notchpeak CHPC

## Distributed CPU

| Threads            | 1      | 2      | 4     | 8     | 16    | 32    |
|--------------------|--------|--------|-------|-------|-------|-------|
| Time to converge   | 2556ms | 1281ms | 640ms | 347ms | 180ms | 171ms |

## Distributed GPU
Due to the nature of the distributed GPU implementation, adding more threads on a single machine will harm performance. As such, different block sizes have been compared.

1024 is the max block size for most modern NVIDIA cards.

| Block size       | 1      | 2      | 4      | 8      | 16     | 32     | 64     | 128    | 256    | 512    | 1024    |
|------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
| Time to converge | 1689ms | 1823ms | 2229ms | 3026ms | 2973ms | 4610ms | 7490ms | 7646ms | 7165ms | 8551ms | 13927ms |