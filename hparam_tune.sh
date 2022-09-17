(
python at_cooling.py --device cuda:1 --cooling-ratio 0 --cooling-interval 201 --cooling-start-epoch 201
python at_cooling.py --device cuda:1 --cooling-ratio 0.1 --cooling-interval 10 --cooling-start-epoch 1
)&
(
python at_cooling.py --device cuda:2 --cooling-ratio 0.2 --cooling-interval 10 --cooling-start-epoch 1
python at_cooling.py --device cuda:2 --cooling-ratio 0.3 --cooling-interval 10 --cooling-start-epoch 1
)&
