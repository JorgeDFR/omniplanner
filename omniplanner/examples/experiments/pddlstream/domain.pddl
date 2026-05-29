(define (domain pick-and-place)
  (:requirements :strips :adl :derived-predicates)

  (:predicates
    (place ?p)
    (dsg-object ?o)
    (region ?r)

    (at-poi ?p)
    (object-in-place ?o ?p)
    (place-in-region ?p ?r)

    (connected ?p1 ?p2)

    (at-place ?p)
    (at-object ?o)
    (in-region ?r)

    (visited-place ?p)
    (visited-object ?o)
    (visited-region ?r)

    (hand-full)
    (suspicious ?o)

    (holding ?o)
    (safe ?o)
  )

  (:functions
    (distance ?p1 ?p2)
    (total-cost)
  )

  (:derived (at-place ?p)
    (and (place ?p) (at-poi ?p)))

  (:derived (at-object ?o)
    (exists (?p) (and (place ?p) (dsg-object ?o)
                      (at-poi ?p)
                      (object-in-place ?o ?p))))

  (:derived (in-region ?r)
    (exists (?p) (and (place ?p) (region ?r)
                      (at-poi ?p)
                      (place-in-region ?p ?r))))

  (:derived (visited-object ?o)
    (exists (?p) (and (place ?p) (dsg-object ?o)
                      (visited-place ?p)
                      (object-in-place ?o ?p))))

  (:derived (visited-region ?r)
    (exists (?p) (and (place ?p) (region ?r)
                      (visited-place ?p)
                      (place-in-region ?p ?r))))

  (:derived (safe ?o)
    (and (dsg-object ?o) (not (suspicious ?o))))

  (:action move
    :parameters (?p1 ?p2)
    :precondition (and (at-poi ?p1)
                       (connected ?p1 ?p2))
    :effect (and (not (at-poi ?p1))
                 (at-poi ?p2)
                 (visited-place ?p2)
                 (increase (total-cost) (distance ?p1 ?p2)))
  )

  (:action pick-object
    :parameters (?o ?p)
    :precondition (and (not (hand-full))
                       (safe ?o)
                       (at-object ?o)
                       (object-in-place ?o ?p))
    :effect (and (holding ?o)
                 (hand-full)
                 (not (object-in-place ?o ?p))))

  (:action place-object
    :parameters (?o ?p)
    :precondition (and (holding ?o)
                       (at-poi ?p))
    :effect (and (not (holding ?o))
                 (not (hand-full))
                 (object-in-place ?o ?p)))

  (:action inspect
    :parameters (?o)
    :precondition (at-object ?o)
    :effect (and (not (suspicious ?o))
                 (increase (total-cost) 1)))
)
