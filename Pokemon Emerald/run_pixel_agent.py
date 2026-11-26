"""Keep PyTorch out of the spawned CPU emulator processes on Windows."""
if __name__=="__main__":
    from pixel_dreamer import main
    main()
