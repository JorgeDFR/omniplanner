(define (domain test-domain)
    (:requirements :derived-predicates :typing :adl)
    (:types
        place dsg_object region - object
    )

    (:predicates
        (at-poi ?p - place)
        (connected ?s - place ?t - place)
        (suspicious ?o - dsg_object)
        (at-place ?p)
        (at-object ?o ?p)
        (in-region ?r ?p)

        (holding ?o - dsg_object)
        (hand-full)
        (object-in-place ?o - dsg_object ?p - place)
        (place-in-region ?p - place ?r - region)

        (visited-place ?p - place)
        (visited-object ?o - dsg_object)
        (visited-region ?r - region)

        (safe ?o)
    )

    (:functions
        (distance ?s ?t)
        (total-cost)
    )

    (:derived (at-place ?p - place)
        (at-poi ?p))

    (:derived (at-object ?o - dsg_object ?p - place)
        (and (at-poi ?p) (object-in-place ?o ?p)))

    (:derived (in-region ?r - region ?p - place)
        (and (at-poi ?p) (place-in-region ?p ?r)))

    (:derived (visited-object ?o - dsg_object)
        (exists (?p - place) (and (visited-place ?p) (object-in-place ?o ?p))))

    (:derived (visited-region ?r - region)
        (exists (?p - place) (and (visited-place ?p) (place-in-region ?p ?r))))

    (:derived (safe ?o - dsg_object)
        (not (suspicious ?o)))

    (:action goto-poi
        :parameters (?s - place ?t - place)
        :precondition (and (at-poi ?s)
                           (or (connected ?s ?t)
                               (connected ?t ?s)))
        :effect (and (not (at-poi ?s))
                     (at-poi ?t)
                     (visited-place ?t)
                     (increase (total-cost) (distance ?s ?t))
        )
    )

    (:action pick-object
     :parameters (?o - dsg_object ?p - place)
     :precondition (and (not (hand-full))
                        (safe ?o)
                        (at-object ?o ?p)
                        (object-in-place ?o ?p))
     :effect (and (holding ?o)
                  (hand-full)
                  (not (object-in-place ?o ?p))))

    (:action place-object
     :parameters (?o - dsg_object ?p - place)
     :precondition (and (holding ?o) (at-poi ?p))
     :effect (and (not (holding ?o))
                  (not (hand-full))
                  (object-in-place ?o ?p)))

    (:action inspect
     :parameters (?o - dsg_object ?p - place)
     :precondition (at-object ?o ?p)
     :effect (and (not (suspicious ?o))
                  (increase (total-cost) 1)
            )
     )

)
