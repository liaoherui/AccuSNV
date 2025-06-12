import runpy

def main():
    runpy.run_module("local_analysis.accusnv_downstream", run_name="__main__")

if __name__ == "__main__":
    main()
