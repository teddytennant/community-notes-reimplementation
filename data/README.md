# The slice

Community Notes publishes its whole rating history every day. The full dump on
2026-09-02 was 231,012,021 ratings across 9 zipped parts, about 42 GB
uncompressed, which is more than this project needs and more than the machine
had room for.

`scripts/build_slice.py` downloads one part at a time, keeps the rows it wants
and deletes the part before fetching the next, so peak disk stays around 3 GB.
The parts are sorted by noteId, which means note creation date, so only part 7
had anything in the window.

What is here, from dump date 2026-09-02:

| file | rows | what it is |
| --- | --- | --- |
| `ratings.parquet` | 1,967,908 | every rating on a note created June 1 to 14, 2026 |
| `notes.parquet` | 46,527 | those notes |
| `status.parquet` | 40,636 | the official status history for them |

The window is two weeks rather than the whole month because a month came to
4.2M ratings and the fit is meant to run on a laptop. June is far enough back
that statuses have settled. `slice.json` records the counts and the columns.

Ratings are mapped to the float the official scorer uses: NOT_HELPFUL 0,
SOMEWHAT_HELPFUL 0.5, HELPFUL 1.

The dump does not publish the fitted intercepts, only statuses, so
`currentCoreStatus` is the ground truth to compare against.

Rebuild with:

    .venv/bin/python scripts/build_slice.py --date 2026-09-02

The parquet files are gitignored. Building them takes about 15 minutes on a
fast connection.
