from hydra_zen import store

fruit_store = store(package="fruit")
apple_store = store(group="apple")
orange_store = store(group="orange")
veggie_store = store(group="veggie")
letuce_store = veggie_store(group="letuce")

a = fruit_store({"x": 1}, name="a", group="fruit")

apple_store({"x": 2}, name="b", group="fruit/apple")

apple_store({"x": 3}, name="c", group="fruit/apple")

orange_store({"x": 4}, name="d", group="fruit/orange")

veggie_store({"x": 5}, name="e", group="veggie")

5
