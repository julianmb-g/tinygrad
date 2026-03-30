# Tinygrad Orchestration Guidelines

- Ensure tests natively invoke `if __name__ == '__main__': unittest.main()`.

- Prevent OOM deadlocks by avoiding `size='enormous'` targets.
